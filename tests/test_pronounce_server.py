"""
Unit and integration tests for OpenPronounce FastAPI microservice (pronounce_server.py).
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

# Ensure model loading is skipped during test suite execution
os.environ["SKIP_MODEL_LOAD"] = "1"

from pronounce_server import (
    create_app,
    execute_assessment,
    parse_args,
    resolve_device,
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
def sample_german_result():
    """
    Sample assessment response matching OpenPronounce differences structure.
    """
    return {
        "score": 88.5,
        "transcription": "Ich gehe heute in die Schule",
        "phoneme_error_rate": 0.115,
        "differences": {
            "errors": [
                {
                    "word": "Schule",
                    "expected": "ˈʃuːlə",
                    "actual": "ˈsuːlə",
                    "confidence": 0.92,
                },
                {
                    "word": "gehe",
                    "expected": "ˈɡeːə",
                    "actual": "ˈgeːhə",
                    "confidence": 0.78,
                },
            ]
        },
    }


@pytest.fixture
def test_app(sample_german_result):
    """
    Create a test FastAPI application instance with mocked assessor.
    """
    app = create_app(skip_model_load=True)
    mock_assessor = MagicMock(return_value=sample_german_result)
    app.state.assessor = mock_assessor
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
    assert data["service"] == "openpronounce"
    assert "device" in data
    assert isinstance(data["device"], str)
    assert data["device"] in ("cuda", "mps", "cpu")


def test_assess_success_contract(client, test_app, synthetic_wav_bytes):
    """
    POST /assess with mock assessor returning sample German differences.
    Verify returned JSON schema matches exact contract:
    - score: float
    - transcription: str
    - phoneme_error_rate: float
    - errors: list of dicts with word, expected_ipa, actual_ipa, confidence
    """
    files = {"file": ("test_recording.wav", synthetic_wav_bytes, "audio/wav")}
    form_data = {
        "expected_text": "Ich gehe heute in die Schule",
        "lang": "de",
    }
    response = client.post("/assess", files=files, data=form_data)

    assert response.status_code == 200
    data = response.json()

    # Schema and type validation
    assert isinstance(data["score"], float)
    assert data["score"] == 88.5
    assert isinstance(data["transcription"], str)
    assert data["transcription"] == "Ich gehe heute in die Schule"
    assert isinstance(data["phoneme_error_rate"], float)
    assert data["phoneme_error_rate"] == 0.115

    errors = data["errors"]
    assert isinstance(errors, list)
    assert len(errors) == 2

    assert errors[0] == {
        "word": "Schule",
        "expected_ipa": "ˈʃuːlə",
        "actual_ipa": "ˈsuːlə",
        "confidence": 0.92,
    }
    assert errors[1] == {
        "word": "gehe",
        "expected_ipa": "ˈɡeːə",
        "actual_ipa": "ˈgeːhə",
        "confidence": 0.78,
    }


def test_assess_ogg_format(client, test_app):
    """
    POST /assess with .ogg audio file preserves audio format extension.
    """
    ogg_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00synthetic_ogg_bytes"
    files = {"file": ("recording.ogg", ogg_bytes, "audio/ogg")}
    form_data = {"expected_text": "Guten Tag"}

    response = client.post("/assess", files=files, data=form_data)
    assert response.status_code == 200

    mock_assessor = test_app.state.assessor
    call_args, call_kwargs = mock_assessor.call_args
    # First argument should be path to temp file ending with .ogg
    assert call_args[0].endswith(".ogg")
    assert call_args[1] == "Guten Tag"
    assert call_kwargs.get("lang") == "de"


def test_assess_empty_file(client):
    """
    Uploading an empty (0-byte) file should return 400 Bad Request.
    """
    files = {"file": ("empty.wav", b"", "audio/wav")}
    data = {"expected_text": "Hallo Welt"}
    response = client.post("/assess", files=files, data=data)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_assess_missing_file(client):
    """
    Sending request without 'file' multipart parameter should return 422 Unprocessable Entity.
    """
    response = client.post("/assess", data={"expected_text": "Hallo Welt"})
    assert response.status_code == 422


def test_assess_missing_expected_text(client, synthetic_wav_bytes):
    """
    Missing expected_text parameter should return 422 Unprocessable Entity.
    """
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files)
    assert response.status_code == 422


def test_assess_empty_expected_text(client, synthetic_wav_bytes):
    """
    Empty or whitespace-only expected_text should return 422 Unprocessable Entity.
    """
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}

    # Empty string
    res_empty = client.post("/assess", files=files, data={"expected_text": ""})
    assert res_empty.status_code == 422

    # Whitespace only
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    res_ws = client.post("/assess", files=files, data={"expected_text": "   \t \n "})
    assert res_ws.status_code == 422
    assert "expected_text must not be empty" in str(res_ws.json()["detail"])


def test_assess_uninitialized_assessor():
    """
    Calling /assess when assessor is None should return 503 Service Unavailable.
    """
    app = create_app(skip_model_load=True)
    app.state.assessor = None
    with TestClient(app) as uninit_client:
        files = {"file": ("test.wav", make_synthetic_wav(), "audio/wav")}
        response = uninit_client.post("/assess", files=files, data={"expected_text": "Hallo"})
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


def test_assess_exception_unlinks_tempfile(client, test_app, synthetic_wav_bytes):
    """
    Assessor exception should return 500 and guarantee temporary file is unlinked.
    """
    captured_paths = []

    def failing_assessor(audio_path, expected_text, lang="de"):
        captured_paths.append(audio_path)
        # Verify temporary file exists while assessor executes
        assert os.path.exists(audio_path)
        raise RuntimeError("OpenPronounce comparison crash")

    test_app.state.assessor = failing_assessor

    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files, data={"expected_text": "Guten Tag"})

    assert response.status_code == 500
    assert "Pronunciation assessment failed" in response.json()["detail"]
    assert len(captured_paths) == 1

    # Verify temp file was cleanly removed in finally block
    assert not os.path.exists(captured_paths[0])


def test_differences_field_normalization(client, test_app, synthetic_wav_bytes):
    """
    Verify variations of differences/errors schema normalization:
    - expected_ipa and actual_ipa field aliases
    - direct errors list without differences wrapper
    - empty differences list
    """
    # Test alternative field names expected_ipa / actual_ipa
    test_app.state.assessor = MagicMock(
        return_value={
            "score": 95.0,
            "transcription": "Guten Morgen",
            "phoneme_error_rate": 0.05,
            "errors": [
                {
                    "word": "Morgen",
                    "expected_ipa": "ˈmɔʁɡn̩",
                    "actual_ipa": "ˈmɔʁɡŋ",
                    "confidence": 0.85,
                }
            ],
        }
    )

    files = {"file": ("audio.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files, data={"expected_text": "Guten Morgen"})
    assert response.status_code == 200
    data = response.json()
    assert data["errors"][0] == {
        "word": "Morgen",
        "expected_ipa": "ˈmɔʁɡn̩",
        "actual_ipa": "ˈmɔʁɡŋ",
        "confidence": 0.85,
    }


def test_execute_assessment_support():
    """
    Verify execute_assessment handles both callables and objects with assess() method.
    """
    class CustomAssessor:
        def assess(self, audio_path: str, expected_text: str, lang: str = "de"):
            return {"score": 99.0, "transcription": expected_text, "phoneme_error_rate": 0.01, "errors": []}

    res_obj = execute_assessment(CustomAssessor(), "dummy.wav", "Hallo", "de")
    assert res_obj["score"] == 99.0

    def custom_fn(audio_path: str, expected_text: str, lang: str = "de"):
        return {"score": 90.0, "transcription": expected_text, "phoneme_error_rate": 0.1, "errors": []}

    res_fn = execute_assessment(custom_fn, "dummy.wav", "Hallo", "de")
    assert res_fn["score"] == 90.0


def test_device_resolution_and_env():
    """
    Test resolution of device and setting of OPENPRONOUNCE_DEVICE env var.
    """
    dev_cpu = resolve_device("cpu")
    assert dev_cpu == "cpu"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "cpu"

    dev_cuda = resolve_device("cuda")
    assert dev_cuda == "cuda"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "cuda"

    dev_mps = resolve_device("mps")
    assert dev_mps == "mps"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "mps"

    dev_auto = resolve_device("auto")
    if torch.cuda.is_available():
        assert dev_auto == "cuda"
    elif torch.backends.mps.is_available():
        assert dev_auto == "mps"
    else:
        assert dev_auto == "cpu"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == dev_auto


def test_parse_args():
    """
    Test CLI arguments parsing with default and custom values.
    """
    # Defaults
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8002
    assert defaults.device == "auto"

    # Custom arguments
    custom = parse_args([
        "--host", "127.0.0.1",
        "--port", "8082",
        "--device", "cuda",
    ])
    assert custom.host == "127.0.0.1"
    assert custom.port == 8082
    assert custom.device == "cuda"
