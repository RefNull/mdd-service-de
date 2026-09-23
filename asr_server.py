"""
Qwen3-ASR Fluency Microservice.

Provides an OpenAI-compatible speech-to-text API (POST /v1/audio/transcriptions)
and health check endpoint (GET /health) wrapping Qwen3-ASR.

Documentation Sources:
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
- https://github.com/QwenLM/Qwen3-ASR

ARCHITECTURAL NOTE:
DO NOT use AutoModelForSpeechSeq2Seq because qwen3_asr is an audio-conditioned
multimodal architecture (Qwen3ASRForConditionalGeneration).
DO NOT use transformers.pipeline("automatic-speech-recognition") either: it cannot force
a language for this model (generate_kwargs={"language": ...} is Whisper-only and raises).
Load AutoProcessor + Qwen3ASRForConditionalGeneration and force the language through
processor.apply_transcription_request(audio, language=...).
"""

import argparse
import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch

logger = logging.getLogger("asr_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODEL_PRESETS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}


def resolve_model_id(model_name: str) -> str:
    """Resolve a model preset alias or return the custom model ID/path directly."""
    stripped = model_name.strip()
    return MODEL_PRESETS.get(stripped.lower(), stripped)


MODEL_ID = resolve_model_id(os.environ.get("MDD_ASR_MODEL", "Qwen3-ASR-0.6B"))
_transcriber = None


def resolve_device_and_dtype(device_name: str = "auto") -> tuple[str, torch.dtype]:
    """Resolve device and torch dtype based on environment and requested device."""
    if device_name == "auto":
        return (
            ("cuda", torch.bfloat16)
            if torch.cuda.is_available()
            else (("mps", torch.bfloat16) if torch.backends.mps.is_available() else ("cpu", torch.float32))
        )
    dev = device_name.lower().strip()
    return (dev, torch.bfloat16) if dev.startswith(("cuda", "mps")) else (dev, torch.float32)


device, dtype = resolve_device_and_dtype()


def get_transcriber():
    """Load Qwen3-ASR once; return a callable (audio_bytes, language) -> text, or None."""
    global _transcriber
    if _transcriber is None:
        if os.environ.get("SKIP_MODEL_LOAD", "").lower() in ("1", "true", "yes"):
            return None
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        from transformers.pipelines.audio_utils import ffmpeg_read

        logger.info("Loading Qwen3-ASR model '%s' on %s (%s)...", MODEL_ID, device, dtype)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        sampling_rate = processor.feature_extractor.sampling_rate

        def run(audio_bytes: bytes, language: Optional[str]) -> str:
            audio = ffmpeg_read(audio_bytes, sampling_rate)
            inputs = processor.apply_transcription_request(audio, language=language).to(device, dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=448)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return processor.decode(new_tokens, return_format="transcription_only")

        _transcriber = run
    return _transcriber


app = FastAPI(
    title="Qwen3-ASR Fluency Service",
    description="FastAPI microservice wrapping Qwen3-ASR for German speech fluency transcription",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "qwen3-asr", "model": MODEL_ID, "device": device}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form("de"),
    model: Optional[str] = Form(None),
):
    """OpenAI-compatible audio transcription endpoint."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    run = get_transcriber()
    if run is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not initialized or is still loading.",
        )

    try:
        text = await asyncio.to_thread(run, content, language.strip() if language and language.strip() else None)
        return {"text": text}
    except Exception as exc:
        logger.error("ASR transcription error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the ASR server."""
    default_model = os.environ.get("MDD_ASR_MODEL", "Qwen3-ASR-0.6B")
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR FastAPI Microservice")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto (cuda -> mps -> cpu), cuda, mps, cpu (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model preset alias or Hugging Face model ID (default: {default_model})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Alias for --model",
    )
    parsed = parser.parse_args(args)
    chosen_model = parsed.model or parsed.model_id or default_model
    parsed.model = chosen_model
    parsed.model_id = resolve_model_id(chosen_model)
    return parsed


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    if cli_args.device != "auto":
        device, dtype = resolve_device_and_dtype(cli_args.device)
    if cli_args.model_id:
        MODEL_ID = cli_args.model_id
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
