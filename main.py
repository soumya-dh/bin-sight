"""Bin-Sight backend.

Run:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""
from dotenv import load_dotenv

load_dotenv()  # must run before the Gemini client is created

import hashlib
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from classify import CATEGORIES, CATEGORY_BY_ID, classify_image, resolve_bins

app = FastAPI(title="Bin-Sight")


@app.exception_handler(Exception)
async def json_errors(request, exc):
    """Any unexpected crash still returns JSON, so the page can show the real reason."""
    import traceback
    from fastapi.responses import JSONResponse

    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Server error: {type(exc).__name__}: {exc}"})
HISTORY_PATH = Path("disposals.json")


def load_history() -> list:
    return json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []


@app.get("/api/bins")
def bins():
    """Every bin type the setup screen can offer."""
    return CATEGORIES


@app.post("/api/classify")
async def classify(image: UploadFile = File(...), bins: str = Form("[]")):
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty image")
    try:
        selection = json.loads(bins)
    except json.JSONDecodeError:
        raise HTTPException(400, "Bad bin selection")
    chosen = resolve_bins(selection if isinstance(selection, list) else [])
    if not chosen:
        raise HTTPException(400, "Select at least one bin first")
    try:
        result = classify_image(data, image.content_type or "image/jpeg", chosen)
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")
    result["image_hash"] = hashlib.sha256(data).hexdigest()
    return result


class DisposeRequest(BaseModel):
    category_id: str
    category_name: str = ""
    image_hash: str
    bin_id: str = "LAB-1"


@app.post("/api/dispose")
def dispose(req: DisposeRequest):
    try:
        from solana_log import log_disposal  # imported here so the app still starts if Solana libs misbehave
    except Exception as e:
        raise HTTPException(500, f"Solana libraries not installed or broken ({e}). Run: python -m pip install solana solders")

    if req.category_id in CATEGORY_BY_ID:
        name = CATEGORY_BY_ID[req.category_id]["name"]
    elif req.category_id.startswith("custom_"):
        name = req.category_name.strip()[:40] or "Specialist waste"
    else:
        raise HTTPException(400, "Unknown bin")

    record = {
        "app": "bin-sight",
        "bin": req.bin_id[:16],
        "cat": req.category_id[:32],
        "img": req.image_hash[:16],  # hash only, never the photo
        "ts": int(time.time()),
    }
    try:
        tx = log_disposal(record)
    except Exception as e:
        msg = str(e) or type(e).__name__
        if "insufficient" in msg.lower() or "no record of a prior credit" in msg.lower() or "AccountNotFound" in msg:
            msg = "The devnet wallet has no SOL. Run `python solana_log.py` and fund the address at https://faucet.solana.com"
        raise HTTPException(502, f"Solana error: {msg}")

    entry = {**record, **tx, "category_name": name}
    history = load_history()
    history.insert(0, entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    return entry


@app.get("/api/history")
def history():
    return load_history()


# Serve the frontend (must be mounted last)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
