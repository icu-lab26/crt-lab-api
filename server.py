"""CRT Lab API — the *validated* Python algorithm (crt.py) exposed as a fast API.

The browser frontend uploads a clip once; the server transcodes it (ffmpeg, so any
phone format incl. iPhone HEVC works), auto-places the ROI, and measures. Moving the
box just calls /measure again — the clip is already cached server-side, so it's quick.

Endpoints:
  GET  /health           -> {"ok": true}
  POST /upload  (file)   -> {clip_id, frame(base64), W, H, roi, crt90, crt80, span, quality, ita, ita_class}
  POST /measure (json)   -> {roi, crt90, crt80, span, quality, ita, ita_class, fps}
"""
import base64
import datetime as dt
import os
import subprocess
import tempfile
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import crt  # the same validated algorithm used by the Space and the laptop batch

# ---- Firebase (Firestore REST + Storage REST, no SDK), project: crt-lab ----
FB_API_KEY = os.getenv("FB_API_KEY", "AIzaSyD3frLdASCtso09_VKIHLuLLOp62TtgKpE")
FB_PROJECT = os.getenv("FB_PROJECT", "crt-lab")
FB_BASE = f"https://firestore.googleapis.com/v1/projects/{FB_PROJECT}/databases/(default)/documents"
FB_COLL = os.getenv("FB_COLL", "measurements")
FB_BUCKET = os.getenv("FB_BUCKET", f"{FB_PROJECT}.appspot.com")

# ---- access passcode: set APP_PASS in the host env (e.g. Render) to turn the gate on.
# Left unset, there is no gate, so first deploy / testing just works. ----
APP_PASS = os.getenv("APP_PASS", "").strip()


def _bad_pass(p):
    return bool(APP_PASS) and str(p or "").strip() != APP_PASS


def _enc(v):
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": "" if v is None else str(v)}


def fb_set(coll, doc_id, data):
    url = f"{FB_BASE}/{coll}/{doc_id}?key={FB_API_KEY}"
    body = {"fields": {k: _enc(v) for k, v in data.items()}}
    r = requests.patch(url, json=body, timeout=20)
    r.raise_for_status()


def web_encode(src, dst):
    """Re-encode to a browser-safe MP4 (yuv420p + faststart) for reviewer playback.
    Analysis still uses crt.transcode's output, so validation is unaffected."""
    subprocess.run(
        [crt.FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
    )


def fb_upload_video(local_path, clip_id):
    obj = urllib.parse.quote(f"clips/{clip_id}.mp4", safe="")
    up = f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o?uploadType=media&name={obj}"
    with open(local_path, "rb") as f:
        r = requests.post(up, data=f.read(), headers={"Content-Type": "video/mp4"}, timeout=180)
    r.raise_for_status()
    token = (r.json().get("downloadTokens") or "").split(",")[0]
    return (f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o/"
            f"{obj}?alt=media&token={token}")


def fb_upload_photo(local_path, clip_id):
    obj = urllib.parse.quote(f"clips/{clip_id}_skin.jpg", safe="")
    up = f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o?uploadType=media&name={obj}"
    with open(local_path, "rb") as f:
        r = requests.post(up, data=f.read(), headers={"Content-Type": "image/jpeg"}, timeout=120)
    r.raise_for_status()
    token = (r.json().get("downloadTokens") or "").split(",")[0]
    return (f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o/"
            f"{obj}?alt=media&token={token}")


def _dec(v):
    if "stringValue" in v:
        return v["stringValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "booleanValue" in v:
        return v["booleanValue"]
    return ""


def fb_list(coll):
    url = f"{FB_BASE}/{coll}?key={FB_API_KEY}&pageSize=300"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    out = []
    for d in r.json().get("documents", []):
        row = {k: _dec(v) for k, v in d.get("fields", {}).items()}
        row["_id"] = d["name"].split("/")[-1]
        out.append(row)
    return out


def fb_get_doc(coll, doc_id):
    url = f"{FB_BASE}/{coll}/{doc_id}?key={FB_API_KEY}"
    r = requests.get(url, timeout=20)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return {k: _dec(v) for k, v in r.json().get("fields", {}).items()}

app = FastAPI(title="Refill API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to your Netlify domain in production if you like
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE = Path(tempfile.gettempdir()) / "crtlab_clips"
CACHE.mkdir(exist_ok=True)


def _display_frame(work, t=0.78):
    cap = cv2.VideoCapture(str(work))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.5, (n / fps) * t) * fps))
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None


def _ita_class(ita):
    if ita is None:
        return ""
    if ita > 55:
        return "very light"
    if ita > 41:
        return "light"
    if ita > 28:
        return "intermediate"
    if ita > 10:
        return "tan"
    if ita > -30:
        return "brown"
    return "dark"


def _jpg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _dims(work):
    cap = cv2.VideoCapture(str(work))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return W, H


def _measure(work, roi):
    ts, A, fps = crt.extract_signal(work, roi)
    r = crt.compute_crt(ts, A)
    ita = crt.compute_ita(work, roi)
    nan = (r is None) or np.isnan(r["crt90"])
    return {
        "roi": [int(roi[0]), int(roi[1]), int(roi[2])],
        "crt90": None if nan else round(float(r["crt90"]), 2),
        "crt80": None if (r is None or np.isnan(r["crt80"])) else round(float(r["crt80"]), 2),
        "span": None if r is None else round(float(r["span"]), 2),
        "quality": "" if r is None else r.get("quality", ""),
        "fps": round(float(fps)),
        "ita": ita,
        "ita_class": _ita_class(ita),
    }


def ita_from_image(bgr, size):
    """ITA from a single still image, using a centred box of side `size`."""
    import math
    H, W = bgr.shape[:2]
    sz = size if size and size > 0 else min(W, H) // 3
    half = max(20, min(int(sz), min(W, H)) // 2)
    cx, cy = W // 2, H // 2
    rgb = cv2.cvtColor(bgr[cy - half:cy + half, cx - half:cx + half], cv2.COLOR_BGR2RGB).astype(float)
    if rgb.size == 0:
        return None
    keep = ~crt.spec_mask(rgb)
    sel = rgb[keep] if keep.sum() > 0.2 * keep.size else rgb.reshape(-1, 3)
    lab = cv2.cvtColor(sel[None, :].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float).reshape(-1, 3)
    L = lab[:, 0].mean() * 100 / 255
    B = lab[:, 2].mean() - 128
    if abs(B) < 1e-6:
        return None
    return round(math.degrees(math.atan((L - 50) / B)), 1)


def _photo_frame(bgr, sz):
    H, W = bgr.shape[:2]
    half = max(20, min(int(sz), min(W, H)) // 2)
    disp = bgr.copy()
    cv2.rectangle(disp, (W // 2 - half, H // 2 - half), (W // 2 + half, H // 2 + half),
                  (31, 146, 224), max(2, W // 200))  # amber box (BGR)
    return _jpg_b64(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))


@app.get("/health")
def health():
    return {"ok": True}


class CheckReq(BaseModel):
    passcode: str = ""


@app.post("/check")
def check(req: CheckReq):
    # used by the lock screen; ok=True means the code is right (or no gate is set)
    return {"ok": not _bad_pass(req.passcode)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), passcode: str = Form("")):
    """Lightweight: just transcode and return a frame. Analysis happens on /measure,
    after the user has positioned the boxes — keeps this step fast for big clips."""
    if _bad_pass(passcode):
        return {"error": "wrong passcode"}
    clip_id = uuid.uuid4().hex
    raw = CACHE / f"{clip_id}_raw"
    work = CACHE / f"{clip_id}.mp4"
    raw.write_bytes(await file.read())
    try:
        crt.transcode(raw, work)
    except Exception as e:
        return {"error": f"could not read clip: {e}"}
    finally:
        raw.unlink(missing_ok=True)

    base = _display_frame(work)
    W, H = _dims(work)
    return {"clip_id": clip_id, "frame": _jpg_b64(base) if base is not None else None,
            "W": W, "H": H, "roi": [W // 2, H // 2, 80]}


class MeasureReq(BaseModel):
    clip_id: str
    cx: int
    cy: int
    size: int = 80
    passcode: str = ""


@app.post("/measure")
def measure(req: MeasureReq):
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    work = CACHE / f"{req.clip_id}.mp4"
    if not work.exists():
        return {"error": "clip not found — please re-upload"}
    W, H = _dims(work)
    size = max(40, min(int(req.size), min(W, H)))
    half = size // 2
    cx = max(half, min(int(req.cx), W - half))
    cy = max(half, min(int(req.cy), H - half))
    return _measure(work, (cx, cy, size))


class ItaReq(BaseModel):
    clip_id: str
    cx: int
    cy: int
    size: int = 80
    passcode: str = ""


@app.post("/ita")
def ita(req: ItaReq):
    """Skin-tone only (no CRT) at a second box — used for the skin-tone reference."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    work = CACHE / f"{req.clip_id}.mp4"
    if not work.exists():
        return {"error": "clip not found — please re-upload"}
    W, H = _dims(work)
    size = max(40, min(int(req.size), min(W, H)))
    half = size // 2
    cx = max(half, min(int(req.cx), W - half))
    cy = max(half, min(int(req.cy), H - half))
    val = crt.compute_ita(work, (cx, cy, size))
    return {"roi": [cx, cy, size], "ita": val, "ita_class": _ita_class(val)}


class CurveReq(BaseModel):
    clip_id: str
    cx: int
    cy: int
    size: int = 80
    passcode: str = ""


@app.post("/curve")
def curve(req: CurveReq):
    """Recovery-curve diagram (a* over time with t0 and CRT markers) at the CRT box."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    work = CACHE / f"{req.clip_id}.mp4"
    if not work.exists():
        return {"error": "clip not found — please re-upload"}
    W, H = _dims(work)
    size = max(40, min(int(req.size), min(W, H)))
    half = size // 2
    cx = max(half, min(int(req.cx), W - half))
    cy = max(half, min(int(req.cy), H - half))
    ts, A, fps = crt.extract_signal(work, (cx, cy, size))
    r = crt.compute_crt(ts, A)
    if r is None:
        return {"error": "no clear refill to plot"}
    out = CACHE / f"{req.clip_id}_curve.png"
    crt.plot_result(ts, A, r, out, "Recovery curve")
    b = base64.b64encode(out.read_bytes()).decode()
    out.unlink(missing_ok=True)
    return {"curve": "data:image/png;base64," + b}


@app.post("/photo")
async def photo(file: UploadFile = File(...), size: int = Form(0), passcode: str = Form("")):
    """Skin-tone from a still photo (fallback when the clip doesn't show enough skin)."""
    if _bad_pass(passcode):
        return {"error": "wrong passcode"}
    pid = "p" + uuid.uuid4().hex
    raw = CACHE / f"{pid}.img"
    raw.write_bytes(await file.read())
    bgr = cv2.imread(str(raw))
    if bgr is None:
        raw.unlink(missing_ok=True)
        return {"error": "could not read photo (try a JPEG/PNG)"}
    H, W = bgr.shape[:2]
    sz = size if size and size > 0 else min(W, H) // 3
    val = ita_from_image(bgr, sz)
    return {"photo_id": pid, "frame": _photo_frame(bgr, sz), "W": W, "H": H,
            "size": sz, "ita": val, "ita_class": _ita_class(val)}


class PhotoItaReq(BaseModel):
    photo_id: str
    size: int = 0
    passcode: str = ""


@app.post("/photo_ita")
def photo_ita(req: PhotoItaReq):
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    raw = CACHE / f"{req.photo_id}.img"
    if not raw.exists():
        return {"error": "photo not found — please re-upload"}
    bgr = cv2.imread(str(raw))
    if bgr is None:
        return {"error": "could not read photo"}
    H, W = bgr.shape[:2]
    sz = max(40, min(int(req.size) if req.size else min(W, H) // 3, min(W, H)))
    val = ita_from_image(bgr, sz)
    return {"frame": _photo_frame(bgr, sz), "W": W, "H": H,
            "size": sz, "ita": val, "ita_class": _ita_class(val)}


class SaveReq(BaseModel):
    clip_id: str
    subject_id: str
    site: str
    rater: str = ""
    cx: int
    cy: int
    size: int = 80
    notes: str = ""
    stopwatch: str = ""
    skin_cx: int = 0
    skin_cy: int = 0
    skin_size: int = 0
    skin_photo_id: str = ""
    skin_photo_size: int = 0
    passcode: str = ""


@app.post("/save")
def save(req: SaveReq):
    """Re-measure at the final box, upload the clip to Firebase Storage, and write a
    Firestore record — same shape as the Space, so review/agreement/CSV all interoperate."""
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    work = CACHE / f"{req.clip_id}.mp4"
    if not work.exists():
        return {"ok": False, "error": "clip not found — re-upload and analyse again"}
    if not str(req.subject_id).strip():
        return {"ok": False, "error": "enter a Subject ID first"}

    W, H = _dims(work)
    size = max(40, min(int(req.size), min(W, H)))
    half = size // 2
    cx = max(half, min(int(req.cx), W - half))
    cy = max(half, min(int(req.cy), H - half))
    m = _measure(work, (cx, cy, size))

    # skin-tone reference: a photo (preferred if supplied) or a 2nd box on the clip
    skin_ita, skin_class, skin_source, skin_photo_url = "", "", "", ""
    if req.skin_photo_id:
        praw = CACHE / f"{req.skin_photo_id}.img"
        if praw.exists():
            pbgr = cv2.imread(str(praw))
            if pbgr is not None:
                v = ita_from_image(pbgr, req.skin_photo_size)
                skin_ita = "" if v is None else v
                skin_class = _ita_class(v)
                skin_source = "photo"
                try:
                    skin_photo_url = fb_upload_photo(praw, req.clip_id)
                except Exception:
                    pass
    elif int(req.skin_size) > 0:
        ss = max(40, min(int(req.skin_size), min(W, H)))
        sh = ss // 2
        scx = max(sh, min(int(req.skin_cx), W - sh))
        scy = max(sh, min(int(req.skin_cy), H - sh))
        v = crt.compute_ita(work, (scx, scy, ss))
        skin_ita = "" if v is None else v
        skin_class = _ita_class(v)
        skin_source = "clip"

    if m["crt90"] is None:
        status = "no signal"
    elif m["crt90"] > 3:
        status = "abnormal"
    else:
        status = "normal"

    storage_url, up_note = "", ""
    play = CACHE / f"{req.clip_id}_web.mp4"
    try:
        try:
            web_encode(work, play)
            up_src = play
        except Exception:
            up_src = work  # fall back to the analysis file if re-encode fails
        storage_url = fb_upload_video(up_src, req.clip_id)
    except Exception as e:
        up_note = f" (video upload failed — check Storage rules/bucket: {e})"
    finally:
        play.unlink(missing_ok=True)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe = "".join(c for c in str(req.subject_id) if c.isalnum() or c in "-_")
    clip_name = f"{safe}_{req.site}_{now.replace(':', '').replace(' ', '_')}.mp4"
    row = {
        "timestamp": now, "rater": req.rater, "subject_id": req.subject_id, "site": req.site,
        "ita_deg": "" if m["ita"] is None else m["ita"], "ita_class": m["ita_class"],
        "skintone_ita_deg": skin_ita, "skintone_ita_class": skin_class,
        "skin_source": skin_source, "skin_photo_url": skin_photo_url,
        "crt90_s": "" if m["crt90"] is None else m["crt90"],
        "crt80_s": "" if m["crt80"] is None else m["crt80"],
        "crt_stopwatch_a": req.stopwatch, "crt_stopwatch_b": "",
        "status": status, "quality": m["quality"],
        "span": "" if m["span"] is None else m["span"],
        "fps": "" if m["fps"] is None else m["fps"], "roi_source": "web",
        "roi_cx": cx, "roi_cy": cy, "roi_size": size,
        "notes": req.notes, "clip_id": req.clip_id,
        "storage_url": storage_url, "clip_name": clip_name,
    }
    try:
        fb_set(FB_COLL, req.clip_id, row)
    except Exception as e:
        return {"ok": False, "error": f"Firestore write failed (check rules): {e}{up_note}",
                "storage_url": storage_url}
    return {"ok": True, "status": status, "crt90": m["crt90"],
            "storage_url": storage_url, "note": up_note}


class ReviewListReq(BaseModel):
    reviewer: str = ""
    passcode: str = ""


@app.post("/clips")
def clips(req: ReviewListReq):
    """List saved clips for blinded review (no algorithm CRT is returned)."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    try:
        meas = fb_list("measurements")
        reads = fb_list("readings")
    except Exception as e:
        return {"error": f"could not list clips: {e}"}
    done = {r["_id"] for r in reads if r.get("reviewer") == req.reviewer}
    items = []
    for m in meas:
        cid = m.get("clip_id") or m["_id"]
        items.append({
            "clip_id": cid,
            "subject_id": m.get("subject_id", ""),
            "site": m.get("site", ""),
            "storage_url": m.get("storage_url", ""),
            "reviewed": f"{cid}__{req.reviewer}" in done,
        })
    return {"clips": items}


class ReadingReq(BaseModel):
    clip_id: str
    reviewer: str
    crt: str = ""
    cant_assess: bool = False
    passcode: str = ""


@app.post("/reading")
def reading(req: ReadingReq):
    """Save a blinded reviewer reading into the readings collection."""
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    if not req.reviewer:
        return {"ok": False, "error": "pick a reviewer"}
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "clip_id": req.clip_id, "reviewer": req.reviewer,
        "crt_reviewer_s": "" if req.cant_assess else req.crt,
        "cant_assess": bool(req.cant_assess), "timestamp": now,
    }
    try:
        fb_set("readings", f"{req.clip_id}__{req.reviewer}", row)
    except Exception as e:
        return {"ok": False, "error": f"save failed: {e}"}
    return {"ok": True}


# ============================ REGISTERED REVIEWERS ============================
class ReviewersSetReq(BaseModel):
    reviewer1: str = ""
    reviewer2: str = ""
    passcode: str = ""


@app.post("/reviewers")
def reviewers_get(req: CheckReq):
    """Return the two registered reviewer names (stored in readings/_config_reviewers)."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    try:
        c = fb_get_doc("readings", "_config_reviewers")
    except Exception as e:
        return {"error": f"could not load: {e}"}
    return {"reviewer1": c.get("reviewer1", ""), "reviewer2": c.get("reviewer2", "")}


@app.post("/reviewers_set")
def reviewers_set(req: ReviewersSetReq):
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    try:
        fb_set("readings", "_config_reviewers",
               {"reviewer1": req.reviewer1.strip(), "reviewer2": req.reviewer2.strip()})
    except Exception as e:
        return {"ok": False, "error": f"save failed: {e}"}
    return {"ok": True}


# ============================ RESULTS / AGREEMENT ============================
def _f(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _agree(a, b):
    a = np.array(a, float); b = np.array(b, float)
    d = a - b
    n = int(len(d))
    md = float(np.mean(d)); sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    out = {
        "n": n,
        "mean_diff": round(md, 3),
        "sd_diff": round(sd, 3),
        "loa_low": round(md - 1.96 * sd, 3),
        "loa_high": round(md + 1.96 * sd, 3),
        "mean_abs_diff": round(float(np.mean(np.abs(d))), 3),
        "within_0_5s": round(100 * float(np.mean(np.abs(d) <= 0.5)), 1),
        "within_1s": round(100 * float(np.mean(np.abs(d) <= 1.0)), 1),
    }
    if n > 1 and np.std(a) > 0 and np.std(b) > 0:
        out["pearson_r"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    return out


def _icc21(x):
    x = np.array(x, float)
    if x.ndim != 2:
        return None
    n, k = x.shape
    if n < 2 or k < 2:
        return None
    grand = x.mean()
    MSR = k * np.sum((x.mean(axis=1) - grand) ** 2) / (n - 1)
    MSC = n * np.sum((x.mean(axis=0) - grand) ** 2) / (k - 1)
    SST = np.sum((x - grand) ** 2)
    SSE = SST - MSR * (n - 1) - MSC * (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom == 0:
        return None
    return round(float((MSR - MSE) / denom), 3)


def _kappa(a, b):
    a = np.array(a); b = np.array(b)
    n = len(a)
    if n == 0:
        return None
    po = float(np.mean(a == b))
    p1 = np.mean(a); q1 = np.mean(b)
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    if pe >= 1:
        return 1.0
    return round(float((po - pe) / (1 - pe)), 3)


class ResultsReq(BaseModel):
    passcode: str = ""


@app.post("/results")
def results(req: ResultsReq):
    """Unblinded analysis: merge measurements + readings, compute agreement."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    try:
        meas = fb_list("measurements")
        reads = fb_list("readings")
    except Exception as e:
        return {"error": f"could not load: {e}"}

    import itertools
    THR = 3.0
    by_clip = {}
    reviewers = set()
    for r in reads:
        cid = r.get("clip_id")
        rev = r.get("reviewer", "")
        if not cid or not rev:
            continue
        reviewers.add(rev)
        val = None if r.get("cant_assess") else _f(r.get("crt_reviewer_s"))
        by_clip.setdefault(cid, {})[rev] = val
    reviewers = sorted(reviewers)

    rows = []
    for m in meas:
        cid = m.get("clip_id") or m["_id"]
        rows.append({
            "clip_id": cid,
            "subject_id": m.get("subject_id", ""),
            "site": m.get("site", ""),
            "skin_class": m.get("skintone_ita_class", ""),
            "algo_crt": _f(m.get("crt90_s")),
            "bedside": _f(m.get("crt_stopwatch_a")),
            "reviews": {rv: by_clip.get(cid, {}).get(rv) for rv in reviewers},
            "n_reviews": sum(1 for rv in reviewers if by_clip.get(cid, {}).get(rv) is not None),
        })

    # inter-rater
    interrater = None
    if len(reviewers) >= 2:
        A, B = [], []
        for cid, revs in by_clip.items():
            vals = {rv: v for rv, v in revs.items() if v is not None}
            for r1, r2 in itertools.combinations(sorted(vals), 2):
                A.append(vals[r1]); B.append(vals[r2])
        if A:
            interrater = _agree(A, B)
            if len(reviewers) == 2:
                r1, r2 = reviewers
                mat = [[by_clip[c].get(r1), by_clip[c].get(r2)] for c in by_clip
                       if by_clip[c].get(r1) is not None and by_clip[c].get(r2) is not None]
                if len(mat) >= 2:
                    interrater["icc"] = _icc21(mat)
                ca = [1 if r[0] > THR else 0 for r in mat]
                cb = [1 if r[1] > THR else 0 for r in mat]
                if ca:
                    interrater["kappa_3s"] = _kappa(ca, cb)

    # reviewer vs algorithm (pooled)
    rev_algo = None
    RA, AA, cr, cax = [], [], [], []
    for m in meas:
        cid = m.get("clip_id") or m["_id"]
        algo = _f(m.get("crt90_s"))
        if algo is None:
            continue
        for rv in reviewers:
            v = by_clip.get(cid, {}).get(rv)
            if v is not None:
                RA.append(v); AA.append(algo)
                cr.append(1 if v > THR else 0); cax.append(1 if algo > THR else 0)
    if RA:
        rev_algo = _agree(RA, AA)
        rev_algo["kappa_3s"] = _kappa(cr, cax)

    # bedside vs algorithm
    bed_algo = None
    BB, BA = [], []
    for m in meas:
        algo = _f(m.get("crt90_s")); bed = _f(m.get("crt_stopwatch_a"))
        if algo is not None and bed is not None:
            BB.append(bed); BA.append(algo)
    if BB:
        bed_algo = _agree(BB, BA)

    summary = {
        "n_clips": len(meas),
        "n_with_algo": sum(1 for m in meas if _f(m.get("crt90_s")) is not None),
        "reviewers": reviewers,
        "n_reviewed_by": {rv: sum(1 for row in rows if row["reviews"].get(rv) is not None) for rv in reviewers},
        "threshold_s": THR,
        "interrater": interrater,
        "reviewer_vs_algo": rev_algo,
        "bedside_vs_algo": bed_algo,
    }
    return {"reviewers": reviewers, "rows": rows, "summary": summary}


@app.post("/reencode")
def reencode(req: ResultsReq):
    """Repair already-stored clips: re-encode each to browser-safe MP4 and re-upload."""
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    try:
        meas = fb_list("measurements")
    except Exception as e:
        return {"ok": False, "error": f"could not list: {e}"}
    fixed = failed = 0
    for m in meas:
        cid = m.get("clip_id") or m["_id"]
        url = m.get("storage_url", "")
        if not url:
            continue
        raw = CACHE / f"{cid}_dl.mp4"
        web = CACHE / f"{cid}_rw.mp4"
        try:
            rr = requests.get(url, timeout=120)
            rr.raise_for_status()
            raw.write_bytes(rr.content)
            web_encode(raw, web)
            new_url = fb_upload_video(web, cid)
            row = {k: v for k, v in m.items() if k != "_id"}
            row["storage_url"] = new_url
            fb_set(FB_COLL, cid, row)
            fixed += 1
        except Exception:
            failed += 1
        finally:
            raw.unlink(missing_ok=True)
            web.unlink(missing_ok=True)
    return {"ok": True, "fixed": fixed, "failed": failed}


def _ensure_cached(clip_id, storage_url):
    """Make sure the clip is on local disk (download from Storage if needed)."""
    work = CACHE / f"{clip_id}.mp4"
    if work.exists():
        return work
    if not storage_url:
        return None
    rr = requests.get(storage_url, timeout=120)
    rr.raise_for_status()
    work.write_bytes(rr.content)
    return work


class CurveSavedReq(BaseModel):
    clip_id: str
    passcode: str = ""


@app.post("/curve_saved")
def curve_saved(req: CurveSavedReq):
    """Regenerate the recovery curve for an already-saved clip (downloads if needed)."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    try:
        rec = fb_get_doc("measurements", req.clip_id)
    except Exception as e:
        return {"error": f"could not load record: {e}"}
    if not rec:
        return {"error": "record not found"}
    try:
        work = _ensure_cached(req.clip_id, rec.get("storage_url", ""))
    except Exception as e:
        return {"error": f"could not fetch clip: {e}"}
    if work is None or not work.exists():
        return {"error": "clip file not available"}
    try:
        if rec.get("roi_size"):
            roi = (int(rec["roi_cx"]), int(rec["roi_cy"]), int(rec["roi_size"]))
        else:
            roi = crt.auto_roi(work)
        ts, A, fps = crt.extract_signal(work, roi)
        r = crt.compute_crt(ts, A)
        if r is None:
            return {"error": "no clear refill to plot"}
        out = CACHE / f"{req.clip_id}_curve.png"
        crt.plot_result(ts, A, r, out, "Recovery curve")
        b = base64.b64encode(out.read_bytes()).decode()
        out.unlink(missing_ok=True)
        return {"curve": "data:image/png;base64," + b}
    except Exception as e:
        return {"error": f"could not plot: {e}"}


@app.post("/registry")
def registry(req: CheckReq):
    """Full per-clip registry + which reviewers are still pending."""
    if _bad_pass(req.passcode):
        return {"error": "wrong passcode"}
    try:
        meas = fb_list("measurements")
        reads = fb_list("readings")
        cfg = fb_get_doc("readings", "_config_reviewers")
    except Exception as e:
        return {"error": f"could not load: {e}"}

    registered = [n for n in [cfg.get("reviewer1", ""), cfg.get("reviewer2", "")] if n and n.strip()]
    by_clip = {}
    for r in reads:
        cid = r.get("clip_id"); rev = r.get("reviewer", "")
        if not cid or not rev:
            continue
        by_clip.setdefault(cid, {})[rev] = {
            "crt": "" if r.get("cant_assess") else r.get("crt_reviewer_s", ""),
            "cant": bool(r.get("cant_assess")),
        }

    rows = []
    for m in meas:
        if m.get("_id") == "_config_reviewers":
            continue
        cid = m.get("clip_id") or m["_id"]
        revs = by_clip.get(cid, {})
        reviewed_by = sorted(revs.keys())
        missing = [n for n in registered if n not in revs]
        rows.append({
            "clip_id": cid,
            "subject_id": m.get("subject_id", ""),
            "site": m.get("site", ""),
            "rater": m.get("rater", ""),
            "algo_crt90": m.get("crt90_s", ""),
            "algo_crt80": m.get("crt80_s", ""),
            "span": m.get("span", ""),
            "status": m.get("status", ""),
            "quality": m.get("quality", ""),
            "skin_class": m.get("skintone_ita_class", "") or m.get("ita_class", ""),
            "bedside": m.get("crt_stopwatch_a", ""),
            "notes": m.get("notes", ""),
            "storage_url": m.get("storage_url", ""),
            "has_curve": bool(m.get("storage_url")),
            "timestamp": m.get("timestamp", ""),
            "reviews": {rv: revs[rv]["crt"] for rv in revs},
            "reviewed_by": reviewed_by,
            "missing": missing,
        })
    rows.sort(key=lambda x: (str(x["subject_id"]), str(x["site"])))
    return {"registered": registered, "rows": rows}


# ============================ EDIT A CLIP ============================
class EditReq(BaseModel):
    clip_id: str
    subject_id: Optional[str] = None
    site: Optional[str] = None
    rater: Optional[str] = None
    notes: Optional[str] = None
    bedside: Optional[str] = None
    passcode: str = ""


@app.post("/edit")
def edit_clip(req: EditReq):
    """Edit human-entered fields of a saved record (subject, site, rater, bedside, notes)."""
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    try:
        m = fb_get_doc("measurements", req.clip_id)
    except Exception as e:
        return {"ok": False, "error": f"load failed: {e}"}
    if not m:
        return {"ok": False, "error": "record not found"}
    if req.subject_id is not None:
        m["subject_id"] = req.subject_id.strip()
    if req.site is not None:
        m["site"] = req.site.strip()
    if req.rater is not None:
        m["rater"] = req.rater.strip()
    if req.notes is not None:
        m["notes"] = req.notes
    if req.bedside is not None:
        m["crt_stopwatch_a"] = req.bedside.strip()
    try:
        fb_set("measurements", req.clip_id, m)
    except Exception as e:
        return {"ok": False, "error": f"save failed: {e}"}
    return {"ok": True}


# ============================ DELETE A CLIP ============================
def fb_delete(coll, doc_id):
    url = f"{FB_BASE}/{coll}/{doc_id}?key={FB_API_KEY}"
    r = requests.delete(url, timeout=20)
    if r.status_code not in (200, 404):
        r.raise_for_status()


def fb_delete_object(obj_path):
    obj = urllib.parse.quote(obj_path, safe="")
    url = f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o/{obj}"
    requests.delete(url, timeout=30)  # best-effort; ignore result


class DeleteReq(BaseModel):
    clip_id: str
    passcode: str = ""


@app.post("/delete")
def delete_clip(req: DeleteReq):
    """Permanently delete a clip: Firestore record, Storage files, and its readings."""
    if _bad_pass(req.passcode):
        return {"ok": False, "error": "wrong passcode"}
    if not req.clip_id:
        return {"ok": False, "error": "no clip_id"}
    notes = []
    try:
        fb_delete("measurements", req.clip_id)
    except Exception as e:
        return {"ok": False,
                "error": f"delete blocked — check that Firestore rules allow delete: {e}"}
    for obj in (f"clips/{req.clip_id}.mp4", f"clips/{req.clip_id}_skin.jpg"):
        try:
            fb_delete_object(obj)
        except Exception:
            pass
    try:
        for rd in fb_list("readings"):
            if rd.get("clip_id") == req.clip_id:
                try:
                    fb_delete("readings", rd["_id"])
                except Exception:
                    pass
    except Exception as e:
        notes.append(f"readings: {e}")
    return {"ok": True, "note": "; ".join(notes)}
