"""Robust signal extraction for CRT (experimental).

Does NOT change the validated recovery math (crt.compute_crt). It only improves
the *signal* fed into it:
  1. ROI tracking  - the CRT box follows the spot frame-to-frame (template match),
     so patient/hand movement doesn't drag the box onto the wrong tissue.
  2. Reference normalization - a second box (the skin-tone box) on steady, perfused
     skin is tracked too; its a* fluctuation (common-mode lighting / auto-exposure)
     is subtracted from the CRT box, cancelling light changes that hit both.
Then the same crt.compute_crt() runs on the cleaned signal.
"""
import cv2
import numpy as np
import crt


def _a_mean(rgb):
    keep = ~crt.spec_mask(rgb)
    sel = rgb[keep] if keep.sum() > 0.2 * keep.size else rgb.reshape(-1, 3)
    if sel.size == 0:
        return np.nan
    lab = cv2.cvtColor(sel[None, :].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float).reshape(-1, 3)
    return float(lab[:, 1].mean() - 128)


def _fill_nan(a):
    a = np.asarray(a, float)
    if np.all(np.isnan(a)):
        return np.zeros_like(a)
    m = np.nanmean(a)
    out = a.copy()
    out[np.isnan(out)] = m
    return out


def _track(gray, center, s, template, W, H):
    """Find the template near `center` in this frame; return (new_center, score)."""
    cx, cy = center
    h = s // 2
    win = s  # search radius around the box
    x0, x1 = max(0, cx - h - win), min(W, cx + h + win)
    y0, y1 = max(0, cy - h - win), min(H, cy + h + win)
    region = gray[y0:y1, x0:x1]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return center, 0.0
    res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, mx, _, mloc = cv2.minMaxLoc(res)
    ncx = x0 + mloc[0] + template.shape[1] // 2
    ncy = y0 + mloc[1] + template.shape[1] // 2
    if mx < 0.3:                      # weak match -> keep previous position
        return center, mx
    return [int(ncx), int(ncy)], float(mx)


def _patch(bgr, center, s, W, H):
    cx, cy = center
    h = s // 2
    x0, x1 = max(0, cx - h), min(W, cx + h)
    y0, y1 = max(0, cy - h), min(H, cy + h)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].astype(float)


def _tmpl(gray, center, s, W, H):
    cx, cy = center
    h = s // 2
    x0, x1 = max(0, cx - h), min(W, cx + h)
    y0, y1 = max(0, cy - h), min(H, cy + h)
    return gray[y0:y1, x0:x1]


def measure_robust(video_path, box, ref):
    """box, ref = (cx, cy, size). Returns dict with the recovery result + diagnostics."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, bgr = cap.read()
    if not ok:
        cap.release()
        return None
    H, W = bgr.shape[:2]
    g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tb = _tmpl(g0, box[:2], box[2], W, H)
    tr = _tmpl(g0, ref[:2], ref[2], W, H)
    bc, rc, bs, rs = [box[0], box[1]], [ref[0], ref[1]], box[2], ref[2]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ts, Ab, Ar, disp = [], [], [], []
    prev = None
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if i > 0:
            bc, _ = _track(gray, bc, bs, tb, W, H)
            rc, _ = _track(gray, rc, rs, tr, W, H)
        Ab.append(_a_mean(_patch(bgr, bc, bs, W, H)))
        Ar.append(_a_mean(_patch(bgr, rc, rs, W, H)))
        ts.append(i / fps)
        if prev is not None:
            disp.append(abs(bc[0] - prev[0]) + abs(bc[1] - prev[1]))
        prev = list(bc)
        i += 1
    cap.release()

    ts = np.array(ts)
    Ab = _fill_nan(Ab)
    Ar = _fill_nan(Ar)
    Anorm = Ab - (Ar - np.mean(Ar))            # cancel common-mode lighting wobble
    r = crt.compute_crt(ts, Anorm)
    return {"r": r, "ts": ts, "Anorm": Anorm, "Ab": Ab, "Ar": Ar, "fps": fps,
            "track_disp": float(np.mean(disp)) if disp else 0.0,
            "track_max": float(np.max(disp)) if disp else 0.0}
