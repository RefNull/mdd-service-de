# mdd-service-de

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2FMPS-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast German speech fluency and phoneme-level pronunciation diagnosis service.

## Architecture Overview

`mdd-service-de` provides a dual microservice architecture designed for high-throughput speech assessment, proxied seamlessly by [`llama-swap`](https://github.com/mostlygeek/llama-swap) with an interactive terminal client (`cli.py`):

- **ASR Fluency Service (`asr_server.py`)**: Hugging Face pipeline wrapping `Qwen/Qwen3-ASR-0.6B-hf` for real-time German transcription and fluency verification via an OpenAI-compatible endpoint.
- **Pronunciation Assessment Service (`pronounce_server.py`)**: Headless phonetic aligner wrapping OpenPronounce (`Wav2Vec2` acoustic models + `espeak-ng` phonemizer) computing pronunciation score, Phoneme Error Rate (PER), and word-by-word expected vs. actual IPA transcriptions.
- **Model Router (`llama-swap`)**: On-demand process manager and reverse proxy that loads model backends into VRAM on demand, offloading idle models after inactivity timeouts.
- **Client (`cli.py`)**: Terminal workflow supporting live microphone capture (`sounddevice`) and batch audio evaluation (`--audio-file`).

```
                         +-----------------------------+
                         |      Client (cli.py)        |
                         |  (Microphone / Audio File)  |
                         +--------------+--------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
            [llama-swap]                              [--direct]
                   |                                         |
     +-------------+-------------+             +-------------+-------------+
     |                           |             |                           |
     v                           v             v                           v
+-------------------+   +--------------------+ +-------------------+   +--------------------+
|   ASR Service     |   | Pronounce Service  | |   ASR Service     |   | Pronounce Service  |
|  (asr_server.py)  |   | (pronounce_server) | |  (asr_server.py)  |   | (pronounce_server) |
|    port 8001      |   |     port 8002      | |    port 8001      |   |     port 8002      |
|  Qwen3-ASR-0.6B   |   |   OpenPronounce    | |  Qwen3-ASR-0.6B   |   |   OpenPronounce    |
+-------------------+   +--------------------+ +-------------------+   +--------------------+
```

## Prerequisites

- **Python**: 3.10 or higher
- **System Packages**:
  - `ffmpeg` (required for audio decoding and resampling)
  - `espeak-ng` (required by OpenPronounce for German grapheme-to-phoneme conversion)
- **Hardware Acceleration (Optional, Auto-Detected)**:
  - NVIDIA GPU with CUDA
  - Apple Silicon with MPS
  - Automatic fallback to CPU if no GPU accelerator is detected

### Installing System Dependencies

**Debian / Ubuntu:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
```

**macOS (Homebrew):**
```bash
brew install ffmpeg espeak-ng
```

## Installation

```bash
git clone https://github.com/example/mdd-service-de.git
cd mdd-service-de
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart / Running

### 1. Standalone Microservices

Start each service independently in separate terminals:

```bash
# Terminal 1: ASR Fluency Server (port 8001)
python asr_server.py --host 0.0.0.0 --port 8001

# Terminal 2: Pronunciation Assessment Server (port 8002)
python pronounce_server.py --host 0.0.0.0 --port 8002
```

Both servers support `--device` (`auto`, `cuda`, `mps`, or `cpu`).

### 2. Managed Execution with `llama-swap`

Run both microservices behind `llama-swap` for automated lifecycle and dynamic port assignment:

```yaml
# llama-swap.yaml
models:
  qwen3-asr:
    cmd: python asr_server.py --port ${PORT}
    ttl: 3600
    healthCheckTimeout: 120
    checkEndpoint: /health

  openpronounce:
    cmd: python pronounce_server.py --port ${PORT}
    ttl: 3600
    healthCheckTimeout: 180
    checkEndpoint: /health
```

Launch the `llama-swap` daemon:

```bash
llama-swap --config llama-swap.yaml --port 8080
```

### 3. CLI Usage

The CLI coordinates the two-step evaluation pipeline: first transcribing speech via ASR, then assessing phonetic pronunciation against the reference sentence.

#### Interactive Microphone Capture

```bash
# Using llama-swap proxy (default)
python cli.py --text "Ich habe morgen einen Termin beim Arzt."

# Or querying microservices directly
python cli.py --direct --text "Ich habe morgen einen Termin beim Arzt."
```

Press **Enter** to start recording, speak the sentence, and press **Enter** again to stop.

#### Pre-Recorded Audio File Evaluation

Evaluate an existing `.wav` or `.ogg` audio file:

```bash
python cli.py \
  --audio-file sample.wav \
  --text "Ich habe morgen einen Termin beim Arzt."
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | `"Ich habe morgen einen Termin beim Arzt."` | Target German prompt sentence to evaluate against. |
| `--router-url` | `http://localhost:8080` | `llama-swap` proxy URL. |
| `--audio-file` | `None` | Path to audio file (`.wav` or `.ogg`) for non-interactive evaluation. |
| `--direct` | `False` | Query microservice backends directly without `llama-swap`. |
| `--asr-url` | `http://localhost:8001` | Direct URL for ASR server (when `--direct` is set). |
| `--pronounce-url` | `http://localhost:8002` | Direct URL for Pronounce server (when `--direct` is set). |
| `--samplerate` | `16000` | Microphone capture sample rate in Hz. |

## API Reference

### 1. `POST /v1/audio/transcriptions`
OpenAI-compatible speech-to-text endpoint hosted on `asr_server.py`.

- **Headers**: `Content-Type: multipart/form-data`
- **Parameters**:
  - `file` (binary, required): Audio file (`.wav`, `.ogg`, `.mp3`, etc.).
  - `language` (string, optional, default: `"de"`): ISO language code.
  - `model` (string, optional): Ignored; routed to loaded `Qwen3-ASR` model.
- **Response**:
  ```json
  {
    "text": "Ich habe morgen einen Termin beim Arzt."
  }
  ```

### 2. `POST /assess`
Phoneme-level pronunciation diagnosis endpoint hosted on `pronounce_server.py`.

- **Headers**: `Content-Type: multipart/form-data`
- **Parameters**:
  - `file` (binary, required): Audio file (`.wav` or `.ogg`).
  - `expected_text` (string, required): Expected target text in German.
  - `lang` (string, optional, default: `"de"`): Language code.
- **Response**: See [Output Schema](#output-schema).

### 3. `GET /health`
Service readiness and device health check (available on both microservices).

- **Response (`asr_server.py`)**:
  ```json
  {
    "status": "ok",
    "service": "qwen3-asr",
    "device": "cuda:0"
  }
  ```
- **Response (`pronounce_server.py`)**:
  ```json
  {
    "status": "ok",
    "service": "openpronounce",
    "device": "cuda:0"
  }
  ```

## Output Schema

The `/assess` endpoint returns detailed pronunciation metrics and a breakdown of flagged word phoneme errors:

```json
{
  "score": 82.5,
  "transcription": "ich habe morgen einen termin beim arzt",
  "phoneme_error_rate": 0.175,
  "errors": [
    {
      "word": "termin",
      "expected_ipa": "tɛʁˈmiːn",
      "actual_ipa": "tɛʁˈmɪn",
      "confidence": 0.65
    }
  ]
}
```

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall pronunciation accuracy score (0.0 to 100.0). |
| `transcription` | `string` | OpenPronounce acoustic model transcription. |
| `phoneme_error_rate` | `float` | Phoneme Error Rate (PER) between expected and spoken phonemes. |
| `errors` | `list[object]` | List of flagged words with phonetic mismatches. |
| `errors[].word` | `string` | Flagged word token. |
| `errors[].expected_ipa` | `string` | Canonical German IPA phoneme sequence. |
| `errors[].actual_ipa` | `string` | Spoken IPA phoneme sequence detected from audio. |
| `errors[].confidence` | `float` | Confidence score for the detected phonetic error (0.0 to 1.0). |

## Testing

Run the test suite with `pytest`:

```bash
pytest -v
```

All 45 unit and integration tests run offline using mocks and synthetic in-memory audio without requiring GPU access or downloading heavy model weights.

## License

[MIT](LICENSE)
