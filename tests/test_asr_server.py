"""
Unit and integration tests for Qwen3-ASR FastAPI microservice (asr_server.py).
"""

import io
import os
from pathlib import Path
import sys
import wave
from unittest.mock import MagicMock

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
from fastapi.testclient import TestClient
import torch

# Ensure model downloading is disabled during test suite execution
os.environ["SKIP_MODEL_LOAD"] = "1"

from asr_server import (
    create_app,
    parse_args,
    resolve_device_and_dtype,
)


def make_synthetic_wav(duration_seconds: float = 0.1, sample_rate: int = 16000) -> bytes:
    """
    Generate synthetic 16kHz mono WAV bytes for testing without external audio files.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        num_frames = int(duration_seconds * sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


@pytest.fixture
def synthetic_wav_bytes():
    return make_synthetic_wav()


@pytest.fixture
def test_app():
    """
    Create a test FastAPI application instance with pipeline mocked.
    """
    app = create_app(skip_model_load=True)
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"text": "Ich habe morgen einen Termin beim Arzt."}
    app.state.pipeline = mock_pipeline
    return app


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as tc:
        yield tc


def test_health_endpoint(client):
    """
    GET /health should return 200 OK and expected service metadata.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "qwen3-asr"
    assert "device" in data
    assert isinstance(data["device"], str)
    assert data["device"] in ("cuda", "mps", "cpu")


def test_transcription_success(client, test_app, synthetic_wav_bytes):
    """
    POST /v1/audio/transcriptions should transcribe audio and return standard OpenAI-compatible format.
    """
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data == {"text": "Ich habe morgen einen Termin beim Arzt."}

    # Verify pipeline was called with generate_kwargs={"language": "de"}
    mock_pipe = test_app.state.pipeline
    assert mock_pipe.called
    call_args, call_kwargs = mock_pipe.call_args
    assert len(call_args) == 1
    # Check that temporary file preserved .wav extension
    assert call_args[0].endswith(".wav")
    assert call_kwargs.get("generate_kwargs") == {"language": "de"}


def test_transcription_custom_language(client, test_app, synthetic_wav_bytes):
    """
    POST /v1/audio/transcriptions with custom language parameter.
    """
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    data = {"language": "en"}
    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    mock_pipe = test_app.state.pipeline
    _, call_kwargs = mock_pipe.call_args
    assert call_kwargs.get("generate_kwargs") == {"language": "en"}


def test_transcription_string_return_from_pipeline(client, test_app, synthetic_wav_bytes):
    """
    Verify handling when pipeline returns a raw string rather than a dict.
    """
    test_app.state.pipeline = MagicMock(return_value="Direkter Transkriptionstext")
    files = {"file": ("audio.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    assert response.json() == {"text": "Direkter Transkriptionstext"}


def test_transcription_empty_file(client):
    """
    Uploading an empty (0-byte) file should return 400 Bad Request.
    """
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcription_missing_file(client):
    """
    Sending request without 'file' multipart parameter should return 422 Unprocessable Entity.
    """
    response = client.post("/v1/audio/transcriptions", data={"language": "de"})
    assert response.status_code == 422


def test_transcription_pipeline_not_initialized():
    """
    Calling transcription when pipeline is None should return 503 Service Unavailable.
    """
    app = create_app(skip_model_load=True)
    app.state.pipeline = None
    with TestClient(app) as uninit_client:
        files = {"file": ("test.wav", make_synthetic_wav(), "audio/wav")}
        response = uninit_client.post("/v1/audio/transcriptions", files=files)
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


def test_transcription_pipeline_failure(client, test_app, synthetic_wav_bytes):
    """
    Pipeline failure should return 500 and clean up temporary files.
    """
    test_app.state.pipeline = MagicMock(side_effect=RuntimeError("Audio decoding failed"))
    files = {"file": ("corrupt.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 500
    assert "Transcription failed" in response.json()["detail"]


def test_device_and_dtype_resolution():
    """
    Test resolution of device and torch.dtype.
    """
    # Explicit devices
    dev_cpu, dt_cpu = resolve_device_and_dtype("cpu")
    assert dev_cpu == "cpu"
    assert dt_cpu == torch.float32

    dev_cuda, dt_cuda = resolve_device_and_dtype("cuda")
    assert dev_cuda == "cuda"
    assert dt_cuda == torch.bfloat16

    dev_mps, dt_mps = resolve_device_and_dtype("mps")
    assert dev_mps == "mps"
    assert dt_mps == torch.bfloat16

    # Auto resolution
    dev_auto, dt_auto = resolve_device_and_dtype("auto")
    if torch.cuda.is_available():
        assert dev_auto == "cuda"
        assert dt_auto == torch.bfloat16
    elif torch.backends.mps.is_available():
        assert dev_auto == "mps"
        assert dt_auto == torch.bfloat16
    else:
        assert dev_auto == "cpu"
        assert dt_auto == torch.float32


def test_parse_args():
    """
    Test CLI arguments parsing with default and custom values.
    """
    # Defaults
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8001
    assert defaults.device == "auto"
    assert defaults.model_id == "Qwen/Qwen3-ASR-0.6B-hf"

    # Custom arguments
    custom = parse_args([
        "--host", "127.0.0.1",
        "--port", "8081",
        "--device", "cuda",
        "--model-id", "custom-org/custom-model",
    ])
    assert custom.host == "127.0.0.1"
    assert custom.port == 8081
    assert custom.device == "cuda"
    assert custom.model_id == "custom-org/custom-model"
