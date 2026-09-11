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
Always instantiate via transformers.pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device).
"""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import torch

logger = logging.getLogger("asr_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resolve_device_and_dtype(device_name: str = "auto") -> tuple[str, torch.dtype]:
    """
    Resolve device and torch dtype based on environment and requested device.

    Selection rules:
    - If CUDA is available: device="cuda", dtype=torch.bfloat16.
    - Else if MPS is available: device="mps", dtype=torch.bfloat16.
    - Else: device="cpu", dtype=torch.float32.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            return "mps", torch.bfloat16
        else:
            return "cpu", torch.float32

    # Specific device explicitly specified (e.g., "cuda:0", "mps", "cpu")
    dev = device_name.lower().strip()
    if dev.startswith("cuda"):
        return dev, torch.bfloat16
    elif dev.startswith("mps"):
        return dev, torch.bfloat16
    else:
        return dev, torch.float32


def load_asr_pipeline(model_id: str, device: str, dtype: torch.dtype):
    """
    Load the Hugging Face ASR pipeline for Qwen3-ASR.

    Documentation Sources:
    - https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
    - https://github.com/QwenLM/Qwen3-ASR

    NOTE: DO NOT use AutoModelForSpeechSeq2Seq because qwen3_asr is an audio-conditioned
    multimodal architecture (Qwen3ASRForConditionalGeneration).
    """
    from transformers import pipeline

    logger.info("Initializing transformers ASR pipeline for model '%s' on %s (%s)...", model_id, device, dtype)
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=device,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for FastAPI app.
    Loads the ASR model pipeline on startup unless already injected (e.g. in tests)
    or skipped via skip_model_load flag / SKIP_MODEL_LOAD env var.
    """
    skip = getattr(app.state, "skip_model_load", False) or os.environ.get("SKIP_MODEL_LOAD", "").lower() in ("1", "true", "yes")

    if getattr(app.state, "pipeline", None) is None and not skip:
        app.state.pipeline = load_asr_pipeline(
            model_id=app.state.model_id,
            device=app.state.device,
            dtype=app.state.dtype,
        )
        logger.info("Qwen3-ASR pipeline loaded and ready.")
    elif app.state.pipeline is not None:
        logger.info("Pre-configured ASR pipeline detected, skipping pipeline initialization.")
    else:
        logger.info("Model loading skipped due to skip_model_load flag / SKIP_MODEL_LOAD env var.")

    yield


def create_app(
    model_id: str = "Qwen/Qwen3-ASR-0.6B-hf",
    device: str = "auto",
    skip_model_load: bool = False,
) -> FastAPI:
    """
    Application factory for the ASR FastAPI service.
    """
    resolved_device, resolved_dtype = resolve_device_and_dtype(device)

    app = FastAPI(
        title="Qwen3-ASR Fluency Service",
        description="FastAPI microservice wrapping Qwen3-ASR for German speech fluency transcription",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store configuration in app state
    app.state.model_id = model_id
    app.state.device = resolved_device
    app.state.dtype = resolved_dtype
    app.state.pipeline = None
    app.state.skip_model_load = skip_model_load

    @app.get("/health")
    async def health():
        """
        Health check endpoint.
        Returns {"status": "ok", "service": "qwen3-asr", "device": str(device)}
        """
        return {
            "status": "ok",
            "service": "qwen3-asr",
            "device": str(app.state.device),
        }

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        language: Optional[str] = Form("de"),
        model: Optional[str] = Form(None),
    ):
        """
        OpenAI-compatible audio transcription endpoint.
        Accepts audio file and returns {"text": transcribed_text}.
        """
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        pipe = getattr(app.state, "pipeline", None)
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail="ASR pipeline is not initialized or model is still loading.",
            )

        suffix = Path(file.filename).suffix if file.filename else ""
        if not suffix:
            suffix = ".wav"

        # Preserve file extension suffix (e.g. .wav) for backend audio decoding
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()

            generate_kwargs = {}
            if language and language.strip():
                generate_kwargs["language"] = language.strip()

            # Execute pipeline in a separate thread so as not to block the async event loop
            if generate_kwargs:
                result = await asyncio.to_thread(pipe, tmp.name, generate_kwargs=generate_kwargs)
            else:
                result = await asyncio.to_thread(pipe, tmp.name)

            if isinstance(result, dict):
                transcribed_text = result.get("text", "")
            elif isinstance(result, str):
                transcribed_text = result
            else:
                transcribed_text = str(result)

            return {"text": transcribed_text}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("ASR transcription error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    return app


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the ASR server.
    """
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
        "--model-id",
        type=str,
        default="Qwen/Qwen3-ASR-0.6B-hf",
        help="Hugging Face model ID (default: Qwen/Qwen3-ASR-0.6B-hf)",
    )
    return parser.parse_args(args)


# Module-level app instance for ASGI servers (e.g. uvicorn asr_server:app)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    server_app = create_app(model_id=cli_args.model_id, device=cli_args.device)
    uvicorn.run(server_app, host=cli_args.host, port=cli_args.port)
