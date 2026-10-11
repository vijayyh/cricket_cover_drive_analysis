"""
backend/app.py — CricAI REST API

    uvicorn backend.app:app --port 8000          (run from the project root)

Endpoints
  GET  /api/health              model status
  GET  /api/model               honest evaluation report of the shot classifier
  POST /api/analyze             upload a video or photo -> full analysis JSON
  GET  /api/results             recent analyses (newest first)
  GET  /api/results/{id}        one analysis
  GET  /api/media/{filename}    annotated video / image (supports range requests)

If frontend/dist exists (npm run build), the UI is served from "/".
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.config as config  # noqa: E402
from main import analyze, get_recognizer  # noqa: E402

MAX_UPLOAD_MB = 200

app = FastAPI(title="CricAI — Cricket Shot Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for d in (config.OUTPUT_DIR, config.RESULTS_DIR, config.UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

# MediaPipe graphs and the torch model are not thread-safe; analyses run one at a time.
_analysis_lock = threading.Lock()


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health():
    try:
        rec = get_recognizer()
        return {"status": "ok", "shot_model": "ready", "classes": rec.classes,
                "holdout_accuracy": rec.holdout_accuracy,
                "pose_branch": rec.pose_model is not None}
    except FileNotFoundError as e:
        return {"status": "degraded", "shot_model": str(e)}


@app.get("/api/model")
def model_report():
    if not os.path.exists(config.REPORT_PATH):
        raise HTTPException(404, "No report yet — run: python train_shot_classifier.py")
    return _read_json(config.REPORT_PATH)


@app.post("/api/analyze")
def analyze_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.VIDEO_EXTS | config.IMAGE_EXTS:
        raise HTTPException(415, f"Unsupported file type '{ext or 'unknown'}'. "
                                 "Upload a video (mp4, mov, avi, webm, gif) or a photo (jpg, png).")
    analysis_id = uuid.uuid4().hex[:12]
    upload_path = os.path.join(config.UPLOAD_DIR, analysis_id + ext)
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f, length=1 << 20)
        if os.path.getsize(upload_path) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
        with _analysis_lock:
            try:
                return analyze(upload_path, analysis_id, source_name=file.filename)
            except ValueError as e:
                raise HTTPException(422, str(e))
            except FileNotFoundError as e:
                raise HTTPException(503, str(e))
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


# Backwards-compatible alias for the old frontend
app.post("/upload/")(analyze_upload)


@app.get("/api/results")
def list_results(limit: int = 12):
    files = [os.path.join(config.RESULTS_DIR, f) for f in os.listdir(config.RESULTS_DIR)
             if f.endswith(".json")]
    files.sort(key=os.path.getmtime, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            r = _read_json(p)
            out.append({"id": r["id"], "kind": r["kind"], "source_name": r.get("source_name"),
                        "created": r.get("created"), "shot_type": r["shot"]["shot_type"],
                        "confidence": r["shot"]["confidence"],
                        "grade": (r.get("technique") or {}).get("grade"),
                        "thumbnail": r["media"]["key_frame"]})
        except (KeyError, json.JSONDecodeError):
            continue
    return out


@app.get("/api/results/{analysis_id}")
def get_result(analysis_id: str):
    if not analysis_id.isalnum():
        raise HTTPException(400, "Bad id")
    path = os.path.join(config.RESULTS_DIR, f"{analysis_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Analysis not found")
    return _read_json(path)


@app.get("/api/media/{filename}")
def media(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(config.OUTPUT_DIR, safe)
    if safe != filename or not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)   # Starlette handles HTTP Range for video seeking


_dist = os.path.join(ROOT, "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
