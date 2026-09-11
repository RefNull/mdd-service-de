#!/usr/bin/env python3
"""
Interactive Command-Line Client for German Speech Fluency & Pronunciation Assessment.

Coordinates requests to:
  1. Fluency Microservice (Qwen3-ASR) for real-time transcription verification.
  2. Pronunciation Microservice (OpenPronounce) for phoneme-level German assessment.

Can query either via llama-swap router (default) or directly to individual microservice servers.
Supports microphone capture via sounddevice and pre-recorded audio evaluation via --audio-file.
"""

import argparse
import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


def _extract_port(url: str, default: str) -> str:
    """Extract port number from URL or return default string."""
    try:
        parsed = urlparse(url)
        return str(parsed.port) if parsed.port else default
    except Exception:
        return default


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="German Speech Fluency & Phonetic Pronunciation Assessment Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Ich habe morgen einen Termin beim Arzt.",
        help="Target German prompt sentence to evaluate against.",
    )
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:8080",
        help="llama-swap daemon address.",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to pre-recorded audio file (.wav or .ogg) to evaluate without microphone.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        default=False,
        help="If set, query ASR and Pronounce servers directly instead of through llama-swap.",
    )
    parser.add_argument(
        "--asr-url",
        type=str,
        default="http://localhost:8001",
        help="Direct URL for ASR server (used if --direct is enabled).",
    )
    parser.add_argument(
        "--pronounce-url",
        type=str,
        default="http://localhost:8002",
        help="Direct URL for OpenPronounce server (used if --direct is enabled).",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz for microphone recording.",
    )
    return parser.parse_args(args)


def array_to_wav_bytes(data: np.ndarray, samplerate: int = 16000) -> bytes:
    """
    Pack numpy array of audio samples into an in-memory 16-bit PCM WAV buffer.
    """
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def record_audio(
    samplerate: int = 16000,
    input_fn=input,
) -> bytes:
    """
    Record audio from the microphone until the user presses Enter.
    Returns 16-bit PCM WAV bytes at the specified samplerate.
    """
    input_fn("Press Enter to start recording...")
    chunks: List[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        if status:
            pass
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        with stream:
            input_fn("Recording... Press Enter to stop.\n")
    except Exception as exc:
        raise RuntimeError(
            f"Microphone recording failed: {exc}. "
            "If running in a headless or automated environment, use --audio-file."
        ) from exc

    if not chunks:
        audio_data = np.zeros((0, 1), dtype=np.int16)
    else:
        audio_data = np.concatenate(chunks, axis=0)

    return array_to_wav_bytes(audio_data, samplerate=samplerate)


def load_audio_file(file_path: str | Path) -> bytes:
    """
    Load an audio file into in-memory bytes with validation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")
    return path.read_bytes()


def query_asr(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    language: str = "de",
    filename: str = "audio.wav",
    timeout: float = 60.0,
) -> str:
    """
    Step 1: Send audio to ASR endpoint for fluency transcription.
    """
    url = f"{target_url.rstrip('/')}/v1/audio/transcriptions"
    mime_type = "audio/ogg" if filename.lower().endswith(".ogg") else "audio/wav"
    files = {"file": (filename, wav_bytes, mime_type)}

    data: Dict[str, Any] = {"language": language}
    if not direct:
        data["model"] = "qwen3-asr"

    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
    except requests.exceptions.ConnectionError as err:
        target_name = f"ASR server at {target_url}" if direct else f"llama-swap router at {target_url}"
        port = _extract_port(target_url, "8001")
        guidance = (
            "Ensure asr_server.py is running:\n"
            f"  python asr_server.py --port {port}"
            if direct
            else (
                "Ensure llama-swap is running:\n"
                f"  llama-swap --config llama-swap.yaml\n"
                "Or pass '--direct' to query the ASR and Pronounce servers directly."
            )
        )
        raise ConnectionRefusedError(
            f"Failed to connect to {target_name}.\nConnection was refused.\n{guidance}"
        ) from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"ASR request failed: {err}") from err

    if response.status_code != 200:
        raise RuntimeError(
            f"ASR request returned status {response.status_code}: {response.text}"
        )

    result = response.json()
    return result.get("text", "")


def query_pronounce(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    expected_text: str = "",
    lang: str = "de",
    filename: str = "audio.wav",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Step 2: Send audio and target text to OpenPronounce assessment endpoint.
    """
    if direct:
        url = f"{target_url.rstrip('/')}/assess"
    else:
        url = f"{target_url.rstrip('/')}/upstream/openpronounce/assess"

    mime_type = "audio/ogg" if filename.lower().endswith(".ogg") else "audio/wav"
    files = {"file": (filename, wav_bytes, mime_type)}
    data = {"expected_text": expected_text, "lang": lang}

    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
    except requests.exceptions.ConnectionError as err:
        target_name = f"Pronunciation server at {target_url}" if direct else f"llama-swap router at {target_url}"
        port = _extract_port(target_url, "8002")
        guidance = (
            "Ensure pronounce_server.py is running:\n"
            f"  python pronounce_server.py --port {port}"
            if direct
            else (
                "Ensure llama-swap is running:\n"
                f"  llama-swap --config llama-swap.yaml\n"
                "Or pass '--direct' to query the ASR and Pronounce servers directly."
            )
        )
        raise ConnectionRefusedError(
            f"Failed to connect to {target_name}.\nConnection was refused.\n{guidance}"
        ) from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"Pronunciation assessment request failed: {err}") from err

    if response.status_code != 200:
        raise RuntimeError(
            f"Pronunciation assessment request returned status {response.status_code}: {response.text}"
        )

    return response.json()


def format_score_bar(score: float, width: int = 20) -> str:
    """
    Format score into visual progress bar: e.g. [================>   ] 82.5 / 100
    """
    clamped = max(0.0, min(100.0, float(score)))
    filled = int(round((clamped / 100.0) * width))
    if filled <= 0:
        bar = " " * width
    elif filled >= width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + " " * (width - filled)
    return f"[{bar}] {clamped:.1f} / 100"


def format_report(target_text: str, asr_text: str, assessment: Dict[str, Any]) -> str:
    """
    Generate structured terminal report per PROMPT.md specifications:
      - Overall Score (0-100) with visual score bar
      - Fluency / Transcription Comparison (Target vs ASR vs Wav2Vec2)
      - Phoneme Error Rate (PER)
      - List of flagged words with expected vs actual heard IPA and confidence percentage
    """
    score = float(assessment.get("score", 0.0))
    w2v2_text = assessment.get("transcription", "")
    per = float(assessment.get("phoneme_error_rate", 0.0))
    errors = assessment.get("errors", [])

    score_bar = format_score_bar(score, width=20)
    per_pct = f"{per * 100:.1f}%"

    lines: List[str] = [
        "=" * 74,
        "             GERMAN PRONUNCIATION & FLUENCY ASSESSMENT REPORT",
        "=" * 74,
        f"Overall Score:            {score_bar}",
        f"Phoneme Error Rate (PER): {per_pct} ({per:.3f})",
        "-" * 74,
        "FLUENCY / TRANSCRIPTION COMPARISON:",
        f"  Target Sentence:        {target_text}",
        f"  Qwen3-ASR (Fluency):    {asr_text}",
        f"  Wav2Vec2 (Alignment):   {w2v2_text}",
        "-" * 74,
    ]

    if not errors:
        lines.append("FLAGGED WORDS: None. Perfect pronunciation detected!")
    else:
        lines.append(f"FLAGGED WORDS ({len(errors)} mispronunciation{'s' if len(errors) > 1 else ''} detected):")
        lines.append(f"  {'Word':<16} {'Expected IPA':<18} {'Actual Heard IPA':<18} {'Confidence':<10}")
        lines.append(f"  {'-'*16} {'-'*18} {'-'*18} {'-'*10}")
        for err in errors:
            word = str(err.get("word", ""))
            exp_ipa = str(err.get("expected_ipa") or err.get("expected", ""))
            act_ipa = str(err.get("actual_ipa") or err.get("actual", ""))
            conf = err.get("confidence", 0.0)
            if isinstance(conf, (int, float)):
                conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
            else:
                conf_str = str(conf)
            lines.append(f"  {word:<16} {exp_ipa:<18} {act_ipa:<18} {conf_str:<10}")

    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main client entry point.
    """
    args = parse_args(argv)

    print("\n" + "=" * 74)
    print("         MDD Service DE — Speech Fluency & Pronunciation Client")
    print("=" * 74)
    print(f"Target German Sentence:\n  \"{args.text}\"\n")

    filename = "audio.wav"
    if args.audio_file:
        print(f"Reading pre-recorded audio file: {args.audio_file}")
        try:
            wav_bytes = load_audio_file(args.audio_file)
            filename = Path(args.audio_file).name
        except Exception as exc:
            print(f"\n[Error] Failed to load audio file: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            wav_bytes = record_audio(samplerate=args.samplerate)
        except Exception as exc:
            print(f"\n[Error] Audio recording failed: {exc}", file=sys.stderr)
            return 1

    if not wav_bytes:
        print("\n[Error] Recorded audio buffer is empty.", file=sys.stderr)
        return 1

    asr_endpoint = args.asr_url if args.direct else args.router_url
    pronounce_endpoint = args.pronounce_url if args.direct else args.router_url

    mode = "Direct Server" if args.direct else "llama-swap Router"
    print(f"Coordinating assessment ({mode})...")

    # Step 1: Fluency / ASR
    print("  -> Step 1/2: Querying fluency transcription (Qwen3-ASR)...")
    try:
        asr_text = query_asr(
            wav_bytes=wav_bytes,
            target_url=asr_endpoint,
            direct=args.direct,
            language="de",
            filename=filename,
        )
    except ConnectionRefusedError as exc:
        print(f"\n[Connection Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Error] ASR step failed: {exc}", file=sys.stderr)
        return 1

    # Step 2: Phonetic Assessment
    print("  -> Step 2/2: Querying phonetic assessment (OpenPronounce)...")
    try:
        assessment = query_pronounce(
            wav_bytes=wav_bytes,
            target_url=pronounce_endpoint,
            direct=args.direct,
            expected_text=args.text,
            lang="de",
            filename=filename,
        )
    except ConnectionRefusedError as exc:
        print(f"\n[Connection Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Error] Phonetic assessment step failed: {exc}", file=sys.stderr)
        return 1

    # Terminal report presentation
    print()
    report = format_report(
        target_text=args.text,
        asr_text=asr_text,
        assessment=assessment,
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
