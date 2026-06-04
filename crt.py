"""
CRT measurement — local script.

Usage:
  1. Put this file in any folder (e.g. ~/Desktop/crt_analysis/).
  2. Put your video clips into a subfolder named 'clips/'.
  3. In Terminal, cd into the folder and run:    python3 crt.py
  4. For each clip a window pops up showing a frame.
     Click ONCE on the centre of the pulp (where the slide pressed).
     The window closes automatically and the script measures the CRT.
  5. Results saved to 'results.csv' (open in Excel/Numbers).
     A diagnostic plot per clip saved to 'plots/'.
  6. Drop new clips into 'clips/' and run again — only new ones are processed.
"""

import csv
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np

# -------- paths --------
HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
PLOTS = HERE / "plots"
WORK = HERE / "_work"
CSV_PATH = HERE / "results.csv"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

PLOTS.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

# -------- helpers --------
def transcode(src: Path, dst: Path):
    """Bake rotation + re-encode to H.264 720p so cv2 reads it cleanly."""
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=-2:720", "-c:v", "libx264", "-an", str(dst)],
        check=True,
    )


def spec_mask(rgb):
    """Pixels that are glare (bright + desaturated) — masked out."""
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    return (hsv[..., 2] / 255 > 0.92) & (hsv[..., 1] / 255 < 0.20)


def auto_roi(video_path, size=80, scale=0.25):
    """Locate the ROI automatically: the spot whose a* (redness) recovers most —
    i.e. the re-perfused pulp. Returns (cx, cy, size, confidence) in full-res
    coordinates, or None. `confidence` is the late-minus-early a* rise at that
    spot (a proxy for signal strength; low => weak/dark-skin signal)."""
    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sw, sh = max(1, int(W * scale)), max(1, int(H * scale))
    amaps = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (sw, sh))
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        amaps.append(lab[..., 1] - 128.0)
    cap.release()
    if len(amaps) < 30:
        return None
    arr = np.stack(amaps)                              # (T, sh, sw)
    T = arr.shape[0]
    early = arr[:max(1, int(0.15 * T))].mean(0)
    late = arr[-max(1, int(0.30 * T)):].mean(0)
    diff = late - early                                # recovery amplitude per cell
    skin = late > 2.0                                  # rough skin mask (reddish)
    if skin.sum() < 0.05 * skin.size:
        skin = np.ones_like(skin, bool)                # fallback: whole frame
    sigma = max(1.0, min(sw, sh) * 0.05)
    diffb = cv2.GaussianBlur(np.where(skin, diff, 0).astype(np.float32), (0, 0), sigmaX=sigma)
    diffb = np.where(skin, diffb, -1e9)
    cy_s, cx_s = np.unravel_index(int(np.argmax(diffb)), diffb.shape)
    return (int(cx_s / scale), int(cy_s / scale), size, float(diff[cy_s, cx_s]))


def draw_roi_box(rgb, roi, color=(0, 200, 0)):
    """Return a copy of the RGB frame with the ROI box + center cross drawn."""
    img = rgb.copy()
    cx, cy, s = roi[0], roi[1], roi[2]
    h = s // 2
    cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), color, 3)
    cv2.drawMarker(img, (cx, cy), (220, 0, 0), cv2.MARKER_CROSS, 22, 2)
    return img


def get_roi_from_user(video_path: Path, t_seconds: float):
    """Show a frame at t_seconds and let the user click the pulp centre."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_seconds * fps))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.set_title(f"{video_path.name}\nClick ONCE on the pulp centre (where the slide pressed)")
    ax.axis("off")
    fig.tight_layout()
    clicks = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not clicks:
        return None
    cx, cy = int(clicks[0][0]), int(clicks[0][1])
    return (cx, cy, 80)  # 80x80 px ROI around the click


def extract_signal(video_path: Path, roi):
    """Walk every frame, return time and a* (redness) inside the ROI."""
    cx, cy, size = roi
    half = size // 2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    ts, A = [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].astype(float)
        keep = ~spec_mask(rgb)
        sel = rgb[keep] if keep.sum() > 0.2 * keep.size else rgb.reshape(-1, 3)
        lab = cv2.cvtColor(sel[None, :].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float).reshape(-1, 3)
        A.append(lab[:, 1].mean() - 128)
        ts.append(i / fps)
        i += 1
    cap.release()
    return np.array(ts), np.array(A), fps


def smooth(x, w=7):
    return np.convolve(x, np.ones(w) / w, mode="same")


def compute_crt(ts, A, release_search_s=1.5, perfused_window_s=6.0):
    """Lock onto the FIRST clean refill after release and ignore later movement.

    Recording is expected to start at (or just before) release, so the most-blanched
    moment is found within the first `release_search_s`. The perfused plateau is read
    from a capped window after it, so a late hand movement / slide removal can't be
    mistaken for the refill. Returns the usual fields plus a `quality` flag
    ('ok' | 'low signal' | 'no clear refill' | 'unstable (movement?)')."""
    if len(A) < 15:
        return None
    As = smooth(A, 7)
    fps = len(As) / max(ts[-1], 1e-3)

    # release / blanch: most-blanched moment in the early part of the clip
    early_n = max(3, int(release_search_s * fps))
    bi = int(np.argmin(As[:early_n]))
    blanch = float(As[bi])

    # perfused: robust plateau within a capped window after release (ignores late movement)
    hi = min(len(As), bi + int(perfused_window_s * fps))
    window = As[bi:hi]
    perfused = float(np.percentile(window, 85))
    span = perfused - blanch

    if span < 0.5 or len(window) < 5:
        return dict(t0=float(ts[bi]), bi=bi, blanch=blanch, perfused=perfused,
                    span=span, crt90=float("nan"), crt80=float("nan"),
                    As=As, quality="low signal")

    fr = (As[bi:hi] - blanch) / span
    tt = ts[bi:hi] - ts[bi]

    def cross(p):
        idx = np.where(fr >= p)[0]
        if not len(idx):
            return float("nan")
        j = idx[0]
        if j == 0:
            return 0.0
        return float(tt[j - 1] + (p - fr[j - 1]) * (tt[j] - tt[j - 1]) / (fr[j] - fr[j - 1]))

    crt90, crt80 = cross(0.9), cross(0.8)

    # quality flag
    quality = "ok"
    if span < 3:
        quality = "low signal"
    elif np.isnan(crt90):
        quality = "no clear refill"
    else:
        reached = np.where(fr >= 0.9)[0]
        if len(reached):
            after = fr[reached[0]:]
            if len(after) > 3 and float(np.min(after)) < 0.5:
                quality = "unstable (movement?)"

    return dict(t0=float(ts[bi]), bi=bi, blanch=blanch, perfused=perfused,
                span=span, crt90=crt90, crt80=crt80, As=As, quality=quality)


def compute_ita(video_path, roi):
    """Skin tone as Individual Typology Angle from the perfused ROI: arctan((L*-50)/b*)."""
    import math
    cx, cy, size = roi
    half = size // 2
    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    Ls, As, Bs = [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].astype(float)
        keep = ~spec_mask(rgb)
        sel = rgb[keep] if keep.sum() > 0.2 * keep.size else rgb.reshape(-1, 3)
        lab = cv2.cvtColor(sel[None, :].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float).reshape(-1, 3)
        Ls.append(lab[:, 0].mean() * 100 / 255)
        As.append(lab[:, 1].mean() - 128)
        Bs.append(lab[:, 2].mean() - 128)
    cap.release()
    if len(Ls) < 5:
        return None
    L, A, B = np.array(Ls), np.array(As), np.array(Bs)
    perf = A >= np.percentile(A, 75)
    Lp, Bp = float(L[perf].mean()), float(B[perf].mean())
    if abs(Bp) < 1e-6:
        return None
    return round(math.degrees(math.atan((Lp - 50) / Bp)), 1)


def plot_result(ts, A, r, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, A, alpha=0.3, color="gray", label="raw a*")
    ax.plot(ts, r["As"], lw=2, color="tab:red", label="smoothed a*")
    ax.axvline(ts[r["bi"]], color="black", ls=":", label=f"t0 = {ts[r['bi']]:.2f}s")
    ax.axhline(r["perfused"], color="green", ls="--", alpha=0.5, label=f"perfused {r['perfused']:.1f}")
    ax.axhline(r["blanch"], color="blue", ls="--", alpha=0.5, label=f"blanch {r['blanch']:.1f}")
    if not np.isnan(r["crt90"]):
        ax.axvline(ts[r["bi"]] + r["crt90"], color="orange", ls=":",
                   label=f"CRT90 {r['crt90']:.2f}s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("a* (redness)"); ax.set_title(title)
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH) as f:
        return {row["filename"] for row in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


# -------- main --------
def main():
    if not CLIPS.exists():
        CLIPS.mkdir()
        print(f"Created folder: {CLIPS}")
        print("Drop your video clips into that folder and run me again.")
        return

    clips = sorted(p for p in CLIPS.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".m4v"))
    if not clips:
        print(f"No videos in {CLIPS}.  Drop .mov or .mp4 files there and re-run.")
        return

    done = load_done()
    todo = [c for c in clips if c.name not in done]
    if not todo:
        print(f"All {len(clips)} clip(s) already in results.csv.  Delete results.csv to redo.")
        return

    print(f"Processing {len(todo)} new clip(s)...\n")
    for idx, clip in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {clip.name}")
        try:
            work = WORK / (clip.stem + ".mp4")
            transcode(clip, work)
        except Exception as e:
            print(f"  transcode failed: {e}\n")
            continue

        cap = cv2.VideoCapture(str(work))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        cap.release()

        t_roi = max(0.5, dur * 0.78)  # a frame in the refill window (for overlay / fallback)
        roi4 = auto_roi(work)
        if roi4 is None:
            print("  auto-ROI couldn't lock on (weak signal?) — click the pulp centre")
            roi = get_roi_from_user(work, t_seconds=t_roi)
            if roi is None:
                print("  no ROI, skipping\n")
                continue
            roi_src = "manual"
        else:
            roi = roi4[:3]
            roi_src = "auto"
        print(f"  ROI [{roi_src}] at ({roi[0]},{roi[1]})")

        print("  measuring...")
        ts, A, fps = extract_signal(work, roi)
        r = compute_crt(ts, A)
        if r is None:
            print("  could not compute CRT (clip too short)\n")
            continue
        ita = compute_ita(work, roi)

        plot_path = PLOTS / (clip.stem + ".png")
        plot_result(ts, A, r, plot_path, f"{clip.name}  [{roi_src} ROI]")

        # save an ROI overlay so placement can be eyeballed later
        cap = cv2.VideoCapture(str(work))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_roi * (fps or 30.0)))
        ok, bgr = cap.read()
        cap.release()
        if ok:
            overlay = draw_roi_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), roi)
            cv2.imwrite(str(PLOTS / (clip.stem + "_roi.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        q = r.get("quality", "")
        if np.isnan(r["crt90"]):
            status = "no signal"
        elif r["crt90"] > 3:
            status = "ABNORMAL"
        else:
            status = "normal"

        crt_str = "n/a" if np.isnan(r["crt90"]) else f"{r['crt90']:.2f}s"
        ita_str = "n/a" if ita is None else f"{ita:.1f}°"
        print(f"  CRT90={crt_str}  span={r['span']:.1f}  quality={q}  ITA={ita_str}  [{status}]")
        print(f"  plot → {plot_path.name}\n")

        append_row({
            "filename": clip.name,
            "roi_source": roi_src,
            "roi_cx": roi[0],
            "roi_cy": roi[1],
            "t0_s": f"{r['t0']:.2f}",
            "blanch_a*": f"{r['blanch']:.2f}",
            "perfused_a*": f"{r['perfused']:.2f}",
            "span": f"{r['span']:.2f}",
            "crt90_s": f"{r['crt90']:.3f}" if not np.isnan(r["crt90"]) else "",
            "crt80_s": f"{r['crt80']:.3f}" if not np.isnan(r["crt80"]) else "",
            "quality": q,
            "ita_deg": "" if ita is None else ita,
            "status": status,
            "fps": f"{fps:.0f}",
        })

    print(f"Done. Results: {CSV_PATH}\nPlots:   {PLOTS}\n")


if __name__ == "__main__":
    main()
