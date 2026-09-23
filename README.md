# mdd-service-de (Mispronunciation Detection & Diagnosis)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2FMPS-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MDD** stands for **Mispronunciation Detection and Diagnosis**.

`mdd-service-de` is a fast German speech fluency and phoneme-level pronunciation diagnosis service.

## Architecture Overview

`mdd-service-de` provides a dual microservice architecture designed for high-throughput German speech assessment. It supports single-command combined deployment (`serve.py`), standalone microservices, or dynamic VRAM proxying via [`llama-swap`](https://github.com/mostlygeek/llama-swap):

- **Combined Runner (`serve.py`)**: Recommended single-command supervisor launching and orchestrating both microservices in isolated subprocesses with unified signal handling and exit supervision.
- **ASR Fluency Service (`asr_server.py`)**: Hugging Face pipeline supporting `Qwen3-ASR` presets (0.6B, 1.7B) and custom checkpoints for real-time German transcription via an OpenAI-compatible endpoint (`port 8001`).
- **Pronunciation Assessment Service (`pronounce_server.py`)**: Headless phonetic aligner wrapping OpenPronounce (`Wav2Vec2` acoustic models + `espeak-ng` phonemizer) computing pronunciation scores, Phoneme Error Rates (PER), and word-level expected vs. actual IPA breakdowns (`port 8002`).
- **Dynamic Proxy (`llama-swap`)**: Optional on-demand process manager and reverse proxy that loads model backends into VRAM on demand, offloading idle models after inactivity timeouts (`port 8080`).
- **Clients**: Interactive terminal client (`cli.py`) and external API clients (web frontends, mobile language-learning apps like FreeLingo).

```
                  +---------------------------------------------------+
                  |                      Clients                      |
                  |   Web / Mobile Apps (FreeLingo)  •  CLI (cli.py)  |
                  +-------------------------+-------------------------+
                                            |
                                            v
         +----------------------------------+----------------------------------+
         |                                  |                                  |
         v                                  v                                  v
+-------------------+              +-------------------+              +-------------------+
|  Combined Local   |              |    Standalone     |              |  Managed Router   |
|   (serve.py)      |              |   Microservices   |              |   (llama-swap)    |
+--------+----------+              +----+---------+----+              +---------+---------+
         |                              |         |                             |
         +------------------+           |         |           +-----------------+
                            |           |         |           |
                            v           v         v           v
                    +--------------------+     +--------------------+
                    |    ASR Service     |     | Pronounce Service  |
                    |  (asr_server.py)   |     | (pronounce_server) |
                    |     port 8001      |     |     port 8002      |
                    |--------------------|     |--------------------|
                    | Qwen3-ASR (0.6B,   |     | OpenPronounce      |
                    | 1.7B, or Custom)   |     | (Wav2Vec2 + G2P)   |
                    |                    |     |                    |
                    | /v1/audio/         |     | /assess            |
                    |  transcriptions    |     |                    |
                    +--------------------+     +--------------------+
```

## Prerequisites

- **Python**: 3.10 or higher
- **System Packages**:
  - `ffmpeg` (required for audio decoding and resampling)
  - `espeak-ng` (required by OpenPronounce for German grapheme-to-phoneme conversion)
  - A C compiler and Python headers (`fastdtw`, an OpenPronounce dependency, ships no wheels and is built on install)
- **Hardware Acceleration (Optional, Auto-Detected)**:
  - NVIDIA GPU with CUDA
  - Apple Silicon with MPS
  - Automatic fallback to CPU if no GPU accelerator is detected

### Installing System Dependencies

**Debian / Ubuntu:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng build-essential python3-dev
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

### 1. Combined Local Deployment (`serve.py`) [Recommended]

Launch both microservices together in a single command with unified signal handling and process supervision:

```bash
# Start both services (ASR on 8001, Pronunciation Assessment on 8002)
python serve.py

# Specify model preset (Qwen3-ASR-0.6B or Qwen3-ASR-1.7B)
python serve.py --model Qwen3-ASR-1.7B

# Run with custom ports, hardware device, or custom model checkpoint
python serve.py --host 0.0.0.0 --asr-port 9001 --pronounce-port 9002 --device cuda --model your-org/custom-model
```

Both microservices run in isolated subprocesses to guarantee clean PyTorch CUDA contexts. Pressing `Ctrl+C` cleanly shuts down both processes without leaving orphaned GPU jobs.

### 2. Standalone Microservices

Alternatively, start each service independently in separate terminals:

```bash
# Terminal 1: ASR Fluency Server (port 8001)
python asr_server.py --host 0.0.0.0 --port 8001 --model Qwen3-ASR-0.6B

# Terminal 2: Pronunciation Assessment Server (port 8002)
python pronounce_server.py --host 0.0.0.0 --port 8002
```

Both servers support `--device` (`auto`, `cuda`, `mps`, or `cpu`).

### 3. Managed Execution with `llama-swap`

Run microservices behind [`llama-swap`](https://github.com/mostlygeek/llama-swap) for on-demand VRAM model loading and automatic offloading:

```yaml
# llama-swap.yaml
models:
  # Default 0.6B fluency model
  qwen3-asr:
    cmd: python asr_server.py --port ${PORT} --model Qwen3-ASR-0.6B
    ttl: 3600
    healthCheckTimeout: 120
    checkEndpoint: /health

  # Higher-capacity 1.7B fluency model
  qwen3-asr-1.7b:
    cmd: python asr_server.py --port ${PORT} --model Qwen3-ASR-1.7B
    ttl: 3600
    healthCheckTimeout: 180
    checkEndpoint: /health

  # Pronunciation assessment microservice
  openpronounce:
    cmd: python pronounce_server.py --port ${PORT}
    ttl: 3600
    healthCheckTimeout: 180
    checkEndpoint: /health

# Keep ASR and pronunciation models loaded simultaneously in VRAM
groups:
  pipeline:
    swap: false
    members:
      - qwen3-asr
      - qwen3-asr-1.7b
      - openpronounce
```

Launch the `llama-swap` daemon on a fixed public port (e.g. `8080`):

```bash
llama-swap --config llama-swap.yaml --port 8080
```

#### How Ports Work with `llama-swap`
- **External Client Access (Fixed Port)**: External apps (FreeLingo, web/mobile frontends, `cli.py`) connect to a **single fixed port** on the `llama-swap` daemon (e.g. `http://localhost:8080`). External clients never need to know or track individual microservice ports.
- **Internal Worker Allocation (`${PORT}`)**: Inside `cmd`, `--port ${PORT}` instructs `llama-swap` to automatically allocate an ephemeral loopback port (e.g. `5800`, `5801`) on `127.0.0.1` and route incoming requests there.
- **Optional Static Backend Ports**: If you specifically prefer static backend ports instead of dynamic allocation, you must explicitly declare both `cmd` and `proxy` (e.g. `cmd: ... --port 8001` and `proxy: http://127.0.0.1:8001`). If `proxy` is omitted, `llama-swap` expects `--port ${PORT}`.

#### Why `swap: false` is Required
By default, `llama-swap` enforces mutual model eviction (`swap: true`), unloading the previous model whenever a different model is called. Because speech evaluation is a two-step sequence (Step 1 ASR $\to$ Step 2 Pronunciation), default swapping would evict Qwen3 to load Wav2Vec2 and vice-versa on every recording. Setting `swap: false` in a shared group keeps both models co-resident in VRAM for instant interactive evaluation, unloading them only after the 1-hour inactivity `ttl`.

### 4. CLI Usage

The interactive CLI coordinates the evaluation pipeline: capturing speech from the microphone, transcribing via ASR, and assessing pronunciation against the target sentence.

#### Interactive Microphone Capture

```bash
# Using combined deployment or direct microservices (default ports 8001 / 8002)
python cli.py --direct --text "Ich habe morgen einen Termin beim Arzt."

# Using llama-swap proxy
python cli.py --text "Ich habe morgen einen Termin beim Arzt."
```

Press **Enter** to start recording, speak the sentence, and press **Enter** again to stop.

#### Pre-Recorded Audio File Evaluation

Evaluate an existing `.wav` or `.ogg` audio file:

```bash
python cli.py \
  --direct \
  --audio-file sample.wav \
  --text "Ich habe morgen einen Termin beim Arzt."
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | `"Ich habe morgen einen Termin beim Arzt."` | Target German prompt sentence to evaluate against. |
| `--direct` | `False` | Query microservice backends directly without `llama-swap`. |
| `--asr-url` | `http://localhost:8001` | Direct URL for ASR server (when `--direct` is set). |
| `--pronounce-url` | `http://localhost:8002` | Direct URL for Pronounce server (when `--direct` is set). |
| `--router-url` | `http://localhost:8080` | `llama-swap` proxy URL (when `--direct` is false). |
| `--audio-file` | `None` | Path to audio file (`.wav` or `.ogg`) for non-interactive evaluation. |
| `--samplerate` | `16000` | Microphone capture sample rate in Hz. |

---

## Model Selection

`mdd-service-de` supports standard presets and custom model backends for German ASR:

| Selection | Identifier / Preset | Hugging Face Repository | Description |
|---|---|---|---|
| **0.6B (Default)** | `Qwen3-ASR-0.6B` or `0.6b` | `Qwen/Qwen3-ASR-0.6B-hf` | Lightweight, fast inference, small VRAM footprint. |
| **1.7B** | `Qwen3-ASR-1.7B` or `1.7b` | `Qwen/Qwen3-ASR-1.7B-hf` | High-capacity model with improved transcription accuracy. |
| **CUSTOM** | *Any string* | E.g. `your-org/custom-model` or `/path/to/local/model` | Direct passthrough to Hugging Face `transformers.pipeline`. |

Model selection can be configured in three ways:
1. CLI argument: `python serve.py --model Qwen3-ASR-1.7B` or `python asr_server.py --model Qwen3-ASR-1.7B`
2. Environment variable: `export MDD_ASR_MODEL="Qwen3-ASR-1.7B"`
3. In `llama-swap.yaml`: Route to either `qwen3-asr` or `qwen3-asr-1.7b`.

### Where the weights come from

No weights live in this repository. On first start, `transformers` downloads the resolved
repository from the Hugging Face Hub into the standard cache
(`~/.cache/huggingface/hub/`, relocatable with `HF_HOME` or `HF_HUB_CACHE`); later starts
load from cache. The first launch therefore needs network access and several GB of disk,
and can exceed the `healthCheckTimeout` in `llama-swap.yaml` — warm the cache once with a
direct `python asr_server.py` run. OpenPronounce fetches its own German Wav2Vec2 models the
same way when first used. To run fully offline, pass a local directory as
`--model` and set `HF_HUB_OFFLINE=1`.

---

## Service Communication & API Integration

### Architectural Decoupling

The two microservices are **independent and decoupled**. They do not communicate directly with each other over the network:
- `asr_server.py` performs general speech transcription and fluency validation.
- `pronounce_server.py` performs phonetic alignment, scoring, and phoneme error detection.

Client applications (such as web frontends, mobile language-learning apps like FreeLingo, or backend API gateways) coordinate both services as needed.

### Choosing Your Endpoint URLs

Depending on how `mdd-service-de` is deployed, your application points to either fixed microservice ports or a single `llama-swap` port:

| Endpoint | Direct / Combined (`serve.py`) | Through `llama-swap` (Port 8080) |
|---|---|---|
| **Pronunciation Assessment** | `POST http://localhost:8002/assess` | `POST http://localhost:8080/upstream/openpronounce/assess` |
| **ASR Fluency Transcription** | `POST http://localhost:8001/v1/audio/transcriptions` | `POST http://localhost:8080/v1/audio/transcriptions` *(requires `model` field)* |

### Client Integration Example (TypeScript / Web Frontend)

In a web or mobile language learning client (e.g. FreeLingo):

```typescript
// --- Option A: Direct or Combined Deployment (serve.py) ---
const PRONOUNCE_URL = "http://localhost:8002/assess";
const ASR_URL = "http://localhost:8001/v1/audio/transcriptions";

// --- Option B: Via llama-swap (single fixed port) ---
// const PRONOUNCE_URL = "http://localhost:8080/upstream/openpronounce/assess";
// const ASR_URL = "http://localhost:8080/v1/audio/transcriptions";

// 1. Send recorded audio for pronunciation assessment
async function evaluatePronunciation(audioBlob: Blob, targetSentence: string) {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.wav");
  formData.append("expected_text", targetSentence);
  formData.append("lang", "de");

  const response = await fetch(PRONOUNCE_URL, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Pronunciation assessment failed: ${response.statusText}`);
  }

  const result = await response.json();
  // Returns { score: 82.5, phoneme_error_rate: 0.175, transcription: "...", errors: [...] }
  return result;
}

// 2. Optionally send recorded audio for open-ended ASR transcription
async function transcribeSpeech(audioBlob: Blob, useLlamaSwap: boolean = false) {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.wav");
  formData.append("language", "de");
  if (useLlamaSwap) {
    formData.append("model", "qwen3-asr"); // Required by llama-swap router
  }

  const response = await fetch(ASR_URL, {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  // Returns { text: "Ich habe morgen einen Termin beim Arzt." }
  return result.text;
}
```

### Client Integration Example (Python)

```python
import httpx

# Direct (serve.py / standalone)
PRONOUNCE_URL = "http://localhost:8002/assess"
ASR_URL = "http://localhost:8001/v1/audio/transcriptions"

# Or via llama-swap router:
# PRONOUNCE_URL = "http://localhost:8080/upstream/openpronounce/assess"
# ASR_URL = "http://localhost:8080/v1/audio/transcriptions"

# 1. Assess pronunciation
with open("recording.wav", "rb") as f:
    files = {"file": ("recording.wav", f, "audio/wav")}
    data = {"expected_text": "Ich habe morgen einen Termin beim Arzt.", "lang": "de"}
    response = httpx.post(PRONOUNCE_URL, data=data, files=files)
    assessment = response.json()
    print(f"Score: {assessment['score']}, PER: {assessment['phoneme_error_rate']}")

# 2. Transcribe audio via ASR
with open("recording.wav", "rb") as f:
    files = {"file": ("recording.wav", f, "audio/wav")}
    data = {"language": "de", "model": "qwen3-asr"}
    response = httpx.post(ASR_URL, data=data, files=files)
    transcription = response.json()
    print(f"Transcribed: {transcription['text']}")
```

---

## Architectural Note: Why Not `Qwen3-ForcedAligner`?

It is common to wonder why `mdd-service-de` does not use `Qwen3-ForcedAligner-0.6B` for pronunciation diagnosis.

1. **What Forced Aligners Do**:
   A forced aligner takes audio and a known reference text and outputs start and end timestamps ($\Delta t$) for each character or word. It **assumes the speaker pronounced the text correctly** and merely finds *when* each sound occurred in the audio timeline.
2. **What MDD (Pronunciation Diagnosis) Requires**:
   Mispronunciation Detection and Diagnosis (MDD) requires detecting **what the speaker actually said** versus **what they should have said**:
   - Frame-level acoustic CTC phone posteriors (Wav2Vec2) capture the raw phones actually produced by the speaker without language model correction.
   - Grapheme-to-Phoneme (G2P via `espeak-ng`) derives the canonical German IPA target.
   - Dynamic Time Warping (DTW) and Levenshtein alignment compare canonical IPA with detected IPA, diagnosing substitutions (e.g. `[ɪ]` instead of `[iː]`), deletions, and insertions.
   - Phoneme Error Rate (PER) and word-level confidence scores quantify pronunciation accuracy.

Because a forced aligner cannot detect phoneme deviations or return heard IPA tokens, OpenPronounce's acoustic alignment pipeline is the correct tool for pronunciation diagnosis, while Qwen3-ASR provides natural speech transcription.

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
| `errors[].confidence` | `float` | Highest per-phone error confidence in the word (0.0 to 1.0). `0.0` when OpenPronounce's phone recognizer is disabled. |

## Testing

Run the test suite with `pytest`:

```bash
pytest -v
```

All 65 unit and integration tests run offline using mocks and synthetic in-memory audio without requiring GPU access or downloading heavy model weights.

## License

This project is licensed under the [MIT License](LICENSE).  
For third-party dependencies, models, and system prerequisite licenses, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
