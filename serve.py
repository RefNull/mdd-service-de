#!/usr/bin/env python3
"""Combined local deployment runner for mdd-service-de.

Runs both the Qwen3-ASR microservice and the OpenPronounce microservice
in separate child processes on configurable ports with coordinated lifecycle management.
"""

from __future__ import annotations

import argparse
import logging
import os
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
    asr_cmd, pronounce_cmd = build_commands(args)

    logger.info("Starting combined local deployment...")
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
