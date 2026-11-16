"""Breast Ultrasound AI — web app (FastAPI).

Endpoints:
  GET  /                -> the single-page UI
  POST /api/predict     -> multipart image upload, returns JSON analysis
  GET  /api/report/{id} -> downloadable PDF report for a previous analysis
"""
import base64
import io
import itertools
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from . import inference
from . import report as report_mod

BASE = Path(__file__).resolve().parent.parent
STATIC = BASE / "static"
CHECKPOINT = os.environ.get("MODEL_PATH", str(BASE / "model" / "best_model.pth"))

app = FastAPI(title="Breast Ultrasound AI")

_results = {}
_counter = itertools.count(1)
_store_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    inference.init(CHECKPOINT)


def _b64(img):
    return base64.b64encode(inference.pil_to_png_bytes(img)).decode()


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 15 MB).")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    result = inference.analyse(image)
    rid = f"RPT-{int(time.time())}-{next(_counter):03d}"
    with _store_lock:
        _results[rid] = (result, file.filename or "image")
        if len(_results) > 60:
            _results.pop(next(iter(_results)))
    return JSONResponse({
        "id": rid,
        "filename": file.filename,
        "bbox": result["bbox"],
        "predicted_class": result["predicted_class"],
        "predicted_display": result["predicted_display"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "images": {k: _b64(v) for k, v in result["images"].items()},
    })


@app.get("/api/report/{rid}")
def download_report(rid: str):
    with _store_lock:
        item = _results.get(rid)
    if item is None:
        raise HTTPException(404, "Report not found — please run the analysis again.")
    result, filename = item
    pdf = report_mod.build_report(result, filename, rid)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Ultrasound_Report_{rid}.pdf"'},
    )


from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
