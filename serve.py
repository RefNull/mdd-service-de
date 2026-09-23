#!/usr/bin/env python3
"""Combined local deployment runner for mdd-service-de.

Runs both the Qwen3-ASR microservice and the OpenPronounce microservice
in separate child processes on configurable ports with coordinated lifecycle management.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [serve]: %(message)s",
)
logger = logging.getLogger("serve")


def resolve_python_executable(explicit_exe: Optional[str] = None) -> str:
    """Resolve the Python executable to run child microservices.

    Priority:
    1. Explicitly provided executable (--python flag)
    2. Current sys.executable if running inside an active virtualenv (sys.prefix != sys.base_prefix)
    3. Project .venv or venv Python if present in repository root
    4. Current sys.executable fallback
    """
    if explicit_exe:
        return explicit_exe

    # If running inside an activated virtualenv, stick with it
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return sys.executable

    repo_dir = Path(__file__).resolve().parent
    candidates = [
        repo_dir / ".venv" / "bin" / "python",
        repo_dir / ".venv" / "Scripts" / "python.exe",
        repo_dir / "venv" / "bin" / "python",
        repo_dir / "venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            logger.info("Detected project virtual environment at %s", candidate)
            return str(candidate)

    return sys.executable


def check_dependencies(python_exe: str) -> bool:
    """Verify that the target Python interpreter has the required microservice dependencies."""
    cmd = [python_exe, "-c", "import fastapi; import uvicorn"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            logger.debug("check_dependencies stderr: %s", res.stderr)
            return False
        return True
    except Exception as exc:
        logger.debug("check_dependencies exception: %s", exc)
        return False


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for combined deployment."""
    default_model = os.environ.get("MDD_ASR_MODEL", "Qwen3-ASR-0.6B")

    parser = argparse.ArgumentParser(
        description="Combined Local Deployment for mdd-service-de (ASR + Pronunciation)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind both microservices (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--asr-port",
        type=int,
        default=8001,
        help="Port for Qwen3-ASR microservice (default: 8001)",
    )
    parser.add_argument(
        "--pronounce-port",
        type=int,
        default=8002,
        help="Port for OpenPronounce microservice (default: 8002)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model preset (Qwen3-ASR-0.6B, Qwen3-ASR-1.7B) or custom HF model ID / path (default: {default_model})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto (cuda -> mps -> cpu), cuda, mps, cpu (default: auto)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Path to Python executable for child microservices (defaults to auto-detecting .venv or sys.executable)",
    )

    parsed = parser.parse_args(args)
    if parsed.asr_port == parsed.pronounce_port:
        parser.error(f"--asr-port and --pronounce-port cannot be identical ({parsed.asr_port})")

    return parsed


def build_commands(
    args: argparse.Namespace,
    python_exe: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Construct child command lists for ASR and Pronounce microservices."""
    exe = python_exe or sys.executable
    asr_cmd = [
        exe,
        "asr_server.py",
        "--host",
        str(args.host),
        "--port",
        str(args.asr_port),
        "--device",
        str(args.device),
        "--model",
        str(args.model),
    ]
    pronounce_cmd = [
        exe,
        "pronounce_server.py",
        "--host",
        str(args.host),
        "--port",
        str(args.pronounce_port),
        "--device",
        str(args.device),
    ]
    return asr_cmd, pronounce_cmd


def terminate_process(proc: subprocess.Popen, name: str, timeout: float = 5.0) -> None:
    """Gracefully terminate a child process, falling back to kill if it hangs."""
    if proc.poll() is not None:
        return

    logger.info("Stopping %s (PID %d)...", name, proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        logger.info("%s terminated gracefully.", name)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not exit within %.1fs; sending SIGKILL.", name, timeout)
        proc.kill()
        proc.wait()
    except Exception as exc:
        logger.error("Error stopping %s: %s", name, exc)


def run_combined(args: argparse.Namespace, poll_interval: float = 0.5) -> int:
    """Launch and supervise both microservices until interrupted or a failure occurs."""
    python_exe = resolve_python_executable(getattr(args, "python", None))

    if not check_dependencies(python_exe):
        logger.error(
            "Python interpreter '%s' is missing required packages ('fastapi' or 'uvicorn').\n"
            "  -> Did you forget to activate the virtual environment?\n"
            "     source .venv/bin/activate\n"
            "     python serve.py\n"
            "  -> Or run directly with the virtualenv Python:\n"
            "     .venv/bin/python serve.py\n"
            "  -> If dependencies are not yet installed:\n"
            "     pip install -r requirements.txt",
            python_exe,
        )
        return 1

    asr_cmd, pronounce_cmd = build_commands(args, python_exe=python_exe)

    logger.info("Starting combined local deployment...")
    logger.info("  Python:             %s", python_exe)
    logger.info("  ASR Endpoint:       http://%s:%d (model: %s, device: %s)", args.host, args.asr_port, args.model, args.device)
    logger.info("  Pronounce Endpoint: http://%s:%d (device: %s)", args.host, args.pronounce_port, args.device)

    # Spawn child processes in dedicated subprocesses for clean GPU / memory isolation
    asr_proc = subprocess.Popen(asr_cmd)
    pronounce_proc = subprocess.Popen(pronounce_cmd)

    procs = [("ASR Service", asr_proc), ("Pronounce Service", pronounce_proc)]

    shutdown_initiated = False

    def handle_signal(signum, _frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s signal. Shutting down microservices...", sig_name)
        for name, proc in procs:
            terminate_process(proc, name)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    logger.error("%s exited unexpectedly with code %d", name, ret)
                    # Terminate sibling process
                    for sibling_name, sibling_proc in procs:
                        if sibling_proc is not proc:
                            terminate_process(sibling_proc, sibling_name)
                    return ret
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)
        return 0


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run_combined(cli_args))
