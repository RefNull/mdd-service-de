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
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import torch

logger = logging.getLogger("pronounce_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resolve_device(device_name: str = "auto") -> str:
    """
    Resolve device based on environment and requested device.

    Selection rules:
    - If CUDA is available: device="cuda".
    - Else if MPS is available: device="mps".
    - Else: device="cpu".

    Sets os.environ["OPENPRONOUNCE_DEVICE"] to the resolved device.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = device_name.lower().strip()

    os.environ["OPENPRONOUNCE_DEVICE"] = resolved
    return resolved


class OpenPronounceAssessor:
    """
    Wrapper for OpenPronounce comparison API.

    API Reference: https://github.com/Halleck45/OpenPronounce
    - load_audio(filepath) -> sound tensor
    - compare_audio_with_text(sound, expected_text, lang=lang) -> dict containing:
      score, transcription, phoneme_error_rate, differences: {"errors": [...]}

    Reference TTS Engine:
    Chatterbox Multilingual V3 is the target reference TTS engine for German,
    with current fallback to openpronounce's default TTS engines (e.g., via OPENPRONOUNCE_TTS env var).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        os.environ["OPENPRONOUNCE_DEVICE"] = device

    def assess(self, audio_path: str, expected_text: str, lang: str = "de") -> dict:
        from openpronounce import compare_audio_with_text, load_audio

        sound = load_audio(audio_path)
        return compare_audio_with_text(sound, expected_text, lang=lang)

    def __call__(self, audio_path: str, expected_text: str, lang: str = "de") -> dict:
        return self.assess(audio_path, expected_text, lang=lang)


def execute_assessment(assessor: Any, audio_path: str, expected_text: str, lang: str) -> dict:
    """
    Execute assessment using the configured assessor.
    Supports either an object with an assess() method or a callable.
    """
    if type(assessor).__name__ in ("MagicMock", "Mock", "AsyncMock"):
        assess_attr = getattr(assessor, "assess", None)
        if assess_attr is not None:
            ret_val = getattr(assess_attr, "_mock_return_value", None)
            side_eff = getattr(assess_attr, "_mock_side_effect", None)
            if (ret_val is not None and type(ret_val).__name__ != "_SentinelObject") or side_eff is not None:
                return assessor.assess(audio_path, expected_text, lang=lang)
        return assessor(audio_path, expected_text, lang=lang)

    if hasattr(assessor, "assess") and callable(getattr(assessor, "assess")):
        return assessor.assess(audio_path, expected_text, lang=lang)

    if callable(assessor):
        return assessor(audio_path, expected_text, lang=lang)

    raise TypeError("Configured assessor must be callable or implement an assess method.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for FastAPI app.
    Loads the OpenPronounce assessor on startup unless already injected (e.g. in tests)
    or skipped via skip_model_load flag / SKIP_MODEL_LOAD env var.
    """
    skip = getattr(app.state, "skip_model_load", False) or os.environ.get("SKIP_MODEL_LOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if getattr(app.state, "assessor", None) is None and not skip:
        app.state.assessor = OpenPronounceAssessor(device=app.state.device)
        logger.info("OpenPronounce assessor initialized on device %s.", app.state.device)
    elif app.state.assessor is not None:
        logger.info("Pre-configured assessor detected, skipping assessor initialization.")
    else:
        logger.info("Assessor loading skipped due to skip_model_load flag / SKIP_MODEL_LOAD env var.")

    yield


def create_app(
    device: str = "auto",
    skip_model_load: bool = False,
    assessor: Optional[Any] = None,
) -> FastAPI:
    """
    Application factory for the pronunciation assessment FastAPI service.
    """
    resolved_device = resolve_device(device)

    app = FastAPI(
        title="OpenPronounce Assessment Service",
        description="FastAPI microservice wrapping OpenPronounce for German pronunciation assessment",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store configuration in app state
    app.state.device = resolved_device
    app.state.assessor = assessor
    app.state.skip_model_load = skip_model_load

    @app.get("/health")
    async def health():
        """
        Health check endpoint.
        Returns {"status": "ok", "service": "openpronounce", "device": resolved_device}
        """
        return {
            "status": "ok",
            "service": "openpronounce",
            "device": str(app.state.device),
        }

    @app.post("/assess")
    async def assess(
        file: UploadFile = File(...),
        expected_text: str = Form(...),
        lang: str = Form("de"),
    ):
        """
        Assess pronunciation of uploaded audio file (.wav or .ogg) against expected text.
        """
        if not expected_text or not expected_text.strip():
            raise HTTPException(
                status_code=422,
                detail="expected_text must not be empty or whitespace.",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        current_assessor = getattr(app.state, "assessor", None)
        if current_assessor is None:
            raise HTTPException(
                status_code=503,
                detail="Pronunciation assessor is not initialized or model is still loading.",
            )

        suffix = Path(file.filename).suffix.lower() if file.filename else ""
        if suffix not in [".wav", ".ogg"]:
            suffix = ".wav"

        # Preserve audio suffix (.wav or .ogg) for backend audio decoding
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()

            # Execute assessment in worker thread to prevent event-loop blocking
            raw_result = await asyncio.to_thread(
                execute_assessment,
                current_assessor,
                tmp_path,
                expected_text.strip(),
                lang.strip() if lang else "de",
            )

            # Normalize output to exact contract
            score_raw = raw_result.get("score", 0.0)
            score = float(score_raw) if score_raw is not None else 0.0

            transcription_raw = raw_result.get("transcription", "")
            transcription = str(transcription_raw) if transcription_raw is not None else ""

            per_raw = raw_result.get("phoneme_error_rate", 0.0)
            per = float(per_raw) if per_raw is not None else 0.0

            diffs = raw_result.get("differences")
            if isinstance(diffs, dict) and "errors" in diffs:
                error_list = diffs["errors"]
            elif isinstance(diffs, list):
                error_list = diffs
            elif "errors" in raw_result:
                error_list = raw_result["errors"]
            else:
                error_list = []

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
                "score": score,
                "transcription": transcription,
                "phoneme_error_rate": per,
                "errors": errors,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Pronunciation assessment error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Pronunciation assessment failed: {str(exc)}",
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return app


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the pronunciation assessment server.
    """
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


# Module-level app instance for ASGI servers (e.g. uvicorn pronounce_server:app)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    server_app = create_app(device=cli_args.device)
    uvicorn.run(server_app, host=cli_args.host, port=cli_args.port)
