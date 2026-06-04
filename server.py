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
import tempfile
import urllib.parse
import uuid
from pathlib import Path

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


def fb_upload_video(local_path, clip_id):
    obj = urllib.parse.quote(f"clips/{clip_id}.mp4", safe="")
    up = f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o?uploadType=media&name={obj}"
    with open(local_path, "rb") as f:
        r = requests.post(up, data=f.read(), headers={"Content-Type": "video/mp4"}, timeout=180)
    r.raise_for_status()
    token = (r.json().get("downloadTokens") or "").split(",")[0]
    return (f"https://firebasestorage.googleapis.com/v0/b/{FB_BUCKET}/o/"
            f"{obj}?alt=media&token={token}")

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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...), passcode: str = Form("")):
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
    auto = crt.auto_roi(work)
    if auto is None:
        roi = (W // 2, H // 2, 80)
        res = {"roi": list(roi), "crt90": None, "crt80": None, "span": None,
               "quality": "no clear refill", "fps": None, "ita": None, "ita_class": ""}
    else:
        res = _measure(work, auto[:3])
    res.update({"clip_id": clip_id, "frame": _jpg_b64(base) if base is not None else None,
                "W": W, "H": H})
    return res


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


class SaveReq(BaseModel):
    clip_id: str
    subject_id: str
    site: str
    rater: str = ""
    cx: int
    cy: int
    size: int = 80
    notes: str = ""
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

    if m["crt90"] is None:
        status = "no signal"
    elif m["crt90"] > 3:
        status = "abnormal"
    else:
        status = "normal"

    storage_url, up_note = "", ""
    try:
        storage_url = fb_upload_video(work, req.clip_id)
    except Exception as e:
        up_note = f" (video upload failed — check Storage rules/bucket: {e})"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe = "".join(c for c in str(req.subject_id) if c.isalnum() or c in "-_")
    clip_name = f"{safe}_{req.site}_{now.replace(':', '').replace(' ', '_')}.mp4"
    row = {
        "timestamp": now, "rater": req.rater, "subject_id": req.subject_id, "site": req.site,
        "ita_deg": "" if m["ita"] is None else m["ita"], "ita_class": m["ita_class"],
        "crt90_s": "" if m["crt90"] is None else m["crt90"],
        "crt80_s": "" if m["crt80"] is None else m["crt80"],
        "crt_stopwatch_a": "", "crt_stopwatch_b": "",
        "status": status, "quality": m["quality"],
        "span": "" if m["span"] is None else m["span"],
        "fps": "" if m["fps"] is None else m["fps"], "roi_source": "web",
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
