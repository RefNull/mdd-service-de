"""
OpenPronounce Pronunciation Assessment Microservice.

Provides a pronunciation assessment API (POST /assess) and health check endpoint
(GET /health) wrapping OpenPronounce.

Documentation Sources:
- https://github.com/Halleck45/OpenPronounce

TTS & Reference Audio Architecture:
Chatterbox Multilingual V3 is the target reference TTS engine for German,
with current fallback to openpronounce's default TTS engines (e.g., via
OPENPRONOUNCE_TTS env var).
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import torch

try:
    import openpronounce
except ImportError:
    openpronounce = None

logger = logging.getLogger("pronounce_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resolve_device(device_name: str = "auto") -> str:
    """Resolve device based on environment and requested device."""
    resolved = (
        ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        if device_name == "auto"
        else device_name.lower().strip()
    )
    os.environ["OPENPRONOUNCE_DEVICE"] = resolved
    return resolved


device = resolve_device()

app = FastAPI(
    title="OpenPronounce Assessment Service",
    description="FastAPI microservice wrapping OpenPronounce for German pronunciation assessment",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "openpronounce", "device": device}


@app.post("/assess")
async def assess(
    file: UploadFile = File(...),
    expected_text: str = Form(...),
    lang: str = Form("de"),
):
    """Assess pronunciation of uploaded audio file (.wav or .ogg) against expected text."""
    if not expected_text or not expected_text.strip():
        raise HTTPException(status_code=422, detail="expected_text must not be empty or whitespace.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if openpronounce is None:
        raise HTTPException(
            status_code=503,
            detail="Pronunciation assessor is not initialized or model is still loading.",
        )

    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if suffix not in [".wav", ".ogg"]:
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()

        raw_result = await asyncio.to_thread(
            lambda: openpronounce.compare_audio_with_text(
                openpronounce.load_audio(tmp.name),
                expected_text.strip(),
                lang=lang.strip() if lang else "de",
            )
        )

        diffs = raw_result.get("differences") or raw_result.get("errors", [])
        error_list = diffs.get("errors", []) if isinstance(diffs, dict) else (diffs if isinstance(diffs, list) else [])
        errors = [
            {
                "word": str(err.get("word", "")),
                "expected_ipa": str(err.get("expected") or err.get("expected_ipa", "")),
                "actual_ipa": str(err.get("actual") or err.get("actual_ipa", "")),
                "confidence": float(err.get("confidence", 0.0)),
            }
            for err in error_list
            if isinstance(err, dict)
        ]

        return {
            "score": float(raw_result.get("score") or 0.0),
            "transcription": str(raw_result.get("transcription") or ""),
            "phoneme_error_rate": float(raw_result.get("phoneme_error_rate") or 0.0),
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pronunciation assessment error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pronunciation assessment failed: {str(exc)}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the pronunciation assessment server."""
    parser = argparse.ArgumentParser(description="Run OpenPronounce FastAPI Microservice")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind (default: 8002)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto (cuda -> mps -> cpu), cuda, mps, cpu (default: auto)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    if cli_args.device != "auto":
        device = resolve_device(cli_args.device)
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
