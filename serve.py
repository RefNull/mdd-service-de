#!/usr/bin/env python3
"""Combined local deployment runner for mdd-service-de.

Runs both the Qwen3-ASR microservice and the OpenPronounce microservice
in separate child processes on configurable ports with coordinated lifecycle management.
Supports attached (foreground) mode, detached (background daemon) mode,
automatic virtual environment bootstrapping, and comprehensive service logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Optional
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [serve]: %(message)s",
)
logger = logging.getLogger("serve")


def get_default_pid_file() -> Path:
    """Return default PID file path located at project root."""
    return Path(__file__).resolve().parent / ".serve.pid"


def get_default_log_dir() -> Path:
    """Return default logs directory located at project root."""
    return Path(__file__).resolve().parent / "logs"


def read_pid(pid_file: Path) -> Optional[int]:
    """Read PID integer from file if it exists and is valid."""
    if not pid_file.is_file():
        return None
    try:
        content = pid_file.read_text().strip()
        return int(content) if content else None
    except Exception:
        return None


def is_pid_running(pid: int) -> bool:
    """Check if process with given PID is currently active."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_dependencies(python_exe: str) -> bool:
    """Verify that the target Python interpreter has the required microservice dependencies."""
    cmd = [python_exe, "-c", "import fastapi; import uvicorn"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return res.returncode == 0
    except Exception:
        return False


def bootstrap_environment(repo_dir: Path) -> Optional[str]:
    """Automatically create .venv and install requirements.txt if missing."""
    venv_dir = repo_dir / ".venv"
    venv_python = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )

    if not venv_python.exists():
        logger.info("Project virtual environment not found at %s. Bootstrapping...", venv_dir)
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            logger.info(".venv initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize virtual environment: %s", exc)
            return None

    requirements = repo_dir / "requirements.txt"
    if requirements.exists() and not check_dependencies(str(venv_python)):
        logger.info("Installing dependencies from requirements.txt into .venv (this may take a few minutes)...")
        try:
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements)], check=True)
            logger.info("Dependencies installed successfully into .venv.")
        except Exception as exc:
            logger.error("Failed to install requirements into .venv: %s", exc)
            return None

    if check_dependencies(str(venv_python)):
        return str(venv_python)
    return None


def resolve_python_executable(explicit_exe: Optional[str] = None, auto_setup: bool = True) -> str:
    """Resolve the Python executable to run child microservices.

    Priority:
    1. Explicitly provided executable (--python flag)
    2. Current sys.executable if running inside an active virtualenv (sys.prefix != sys.base_prefix)
    3. Project .venv or venv Python if present in repository root
    4. Auto-bootstrapped .venv if auto_setup is enabled
    5. Current sys.executable fallback
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
            return str(candidate)

    if auto_setup:
        bootstrapped = bootstrap_environment(repo_dir)
        if bootstrapped:
            return bootstrapped

    return sys.executable


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for combined deployment."""
    default_model = os.environ.get("MDD_ASR_MODEL", "Qwen3-ASR-0.6B")

    parser = argparse.ArgumentParser(
        description="Combined Local Deployment for mdd-service-de (ASR + Pronunciation)",
    )
    # Lifecycle action flags
    parser.add_argument(
        "--stop",
        action="store_true",
        default=False,
        help="Stop the currently running background deployment",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Check health and readiness of running microservices",
    )
    parser.add_argument(
        "--logs",
        "-f",
        dest="logs",
        action="store_true",
        default=False,
        help="View or tail recent logs from the deployment",
    )
    parser.add_argument(
        "--detach",
        "-d",
        dest="detach",
        action="store_true",
        default=False,
        help="Run deployment in background as a daemon process",
    )

    # Server configuration
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
    parser.add_argument(
        "--no-auto-setup",
        action="store_true",
        default=False,
        help="Disable automatic virtual environment creation and pip install",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write log files (default: ./logs)",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=None,
        help="Path to PID file for background mode (default: ./.serve.pid)",
    )
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # Internal flag used by --detach runner
    )

    parsed = parser.parse_args(args)
    if not (parsed.stop or parsed.status or parsed.logs):
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
        "-u",
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
        "-u",
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


def handle_stop(pid_file: Path) -> int:
    """Stop the running background deployment."""
    pid = read_pid(pid_file)
    if not pid or not is_pid_running(pid):
        pid_file.unlink(missing_ok=True)
        print("No active combined deployment found.")
        return 0

    print(f"Stopping combined deployment (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        print(f"Process {pid} has already exited.")
        return 0

    deadline = time.time() + 6.0
    while time.time() < deadline:
        if not is_pid_running(pid):
            break
        time.sleep(0.2)

    if is_pid_running(pid):
        print(f"Process {pid} did not terminate within timeout; sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    pid_file.unlink(missing_ok=True)
    print("Combined deployment stopped.")
    return 0


def probe_health(url: str, timeout: float = 2.0) -> Optional[dict]:
    """Probe microservice health check endpoint and return JSON response if successful."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "serve-status"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def handle_status(pid_file: Path, host: str, asr_port: int, pronounce_port: int, log_dir: Path) -> int:
    """Inspect and display status of running microservices."""
    pid = read_pid(pid_file)
    running = pid is not None and is_pid_running(pid)

    bind_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    asr_health = probe_health(f"http://{bind_host}:{asr_port}/health")
    pronounce_health = probe_health(f"http://{bind_host}:{pronounce_port}/health")

    print("=" * 66)
    print("           mdd-service-de Combined Deployment Status")
    print("=" * 66)
    if running:
        print(f"Supervisor Process:   RUNNING (PID {pid})")
    else:
        print("Supervisor Process:   NOT RUNNING")

    if asr_health:
        print(f"ASR Service:          UP (port {asr_port}, model: {asr_health.get('model', 'unknown')}, device: {asr_health.get('device', 'unknown')})")
    elif running:
        print(f"ASR Service:          STARTING / LOADING WEIGHTS (port {asr_port})")
    else:
        print(f"ASR Service:          STOPPED (port {asr_port})")

    if pronounce_health:
        print(f"Pronounce Service:    UP (port {pronounce_port}, device: {pronounce_health.get('device', 'unknown')})")
    elif running:
        print(f"Pronounce Service:    STARTING / LOADING WEIGHTS (port {pronounce_port})")
    else:
        print(f"Pronounce Service:    STOPPED (port {pronounce_port})")

    if log_dir.exists():
        print(f"Log directory:        {log_dir}")
        print(f"                      {log_dir / 'serve.log'} (combined)")
        print(f"                      {log_dir / 'asr.log'}")
        print(f"                      {log_dir / 'pronounce.log'}")
    print("=" * 66)
    return 0 if running else 1


def handle_logs(log_file: Path, follow: bool = False, lines_count: int = 50) -> int:
    """Display or tail deployment log file."""
    if not log_file.is_file():
        print(f"Log file not found: {log_file}")
        print("Start the deployment first with 'python serve.py'.")
        return 1

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
        tail = all_lines[-lines_count:] if len(all_lines) > lines_count else all_lines
        for line in tail:
            sys.stdout.write(line)
        sys.stdout.flush()

        if follow:
            try:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                return 0
    return 0


def spawn_detached(args: argparse.Namespace) -> int:
    """Launch the combined runner in the background as a detached process."""
    pid_file = Path(args.pid_file).resolve() if args.pid_file else get_default_pid_file()
    existing_pid = read_pid(pid_file)
    if existing_pid and is_pid_running(existing_pid):
        print(f"Combined deployment is already running (PID {existing_pid}).")
        print("To stop it, run: python serve.py --stop")
        return 1

    log_dir = Path(args.log_dir).resolve() if args.log_dir else get_default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    serve_log = log_dir / "serve.log"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--host", str(args.host),
        "--asr-port", str(args.asr_port),
        "--pronounce-port", str(args.pronounce_port),
        "--model", str(args.model),
        "--device", str(args.device),
        "--log-dir", str(log_dir),
        "--pid-file", str(pid_file),
    ]
    if args.python:
        cmd.extend(["--python", args.python])
    if args.no_auto_setup:
        cmd.append("--no-auto-setup")

    out_file = open(serve_log, "a", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=out_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    time.sleep(0.6)
    ret = proc.poll()
    if ret is not None:
        print(f"Error: Detached deployment failed to start (exit code {ret}).")
        print(f"Check log for details: {serve_log}")
        return ret

    pid_file.write_text(str(proc.pid))
    print(f"Combined deployment started in background (PID {proc.pid}).")
    print(f"  ASR Endpoint:       http://{args.host}:{args.asr_port} (model: {args.model}, device: {args.device})")
    print(f"  Pronounce Endpoint: http://{args.host}:{args.pronounce_port} (device: {args.device})")
    print(f"  Logs:               {serve_log}")
    print(f"                      {log_dir / 'asr.log'}")
    print(f"                      {log_dir / 'pronounce.log'}")
    print()
    print("Management commands:")
    print("  Check status:       python serve.py --status")
    print("  View logs:          python serve.py --logs (or -f to follow)")
    print("  Stop deployment:    python serve.py --stop")
    return 0


def run_combined(args: argparse.Namespace, poll_interval: float = 0.5) -> int:
    """Launch and supervise both microservices until interrupted or a failure occurs."""
    auto_setup = not getattr(args, "no_auto_setup", False)
    python_exe = resolve_python_executable(getattr(args, "python", None), auto_setup=auto_setup)

    if not check_dependencies(python_exe):
        logger.error(
            "Python interpreter '%s' is missing required packages ('fastapi' or 'uvicorn').\n"
            "  -> To set up the virtual environment automatically, run without --no-auto-setup:\n"
            "     python serve.py\n"
            "  -> Or activate and install manually:\n"
            "     source .venv/bin/activate\n"
            "     pip install -r requirements.txt",
            python_exe,
        )
        return 1

    log_dir = Path(args.log_dir).resolve() if args.log_dir else get_default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path(args.pid_file).resolve() if args.pid_file else get_default_pid_file()
    # Write PID file if running as internal worker
    if getattr(args, "internal_worker", False):
        pid_file.write_text(str(os.getpid()))

    asr_log_path = log_dir / "asr.log"
    pronounce_log_path = log_dir / "pronounce.log"
    serve_log_path = log_dir / "serve.log"

    asr_log_f = open(asr_log_path, "a", buffering=1, encoding="utf-8")
    pronounce_log_f = open(pronounce_log_path, "a", buffering=1, encoding="utf-8")
    serve_log_f = open(serve_log_path, "a", buffering=1, encoding="utf-8")

    asr_cmd, pronounce_cmd = build_commands(args, python_exe=python_exe)

    logger.info("Starting combined local deployment...")
    logger.info("  Python:             %s", python_exe)
    logger.info("  ASR Endpoint:       http://%s:%d (model: %s, device: %s)", args.host, args.asr_port, args.model, args.device)
    logger.info("  Pronounce Endpoint: http://%s:%d (device: %s)", args.host, args.pronounce_port, args.device)
    logger.info("  Log directory:      %s", log_dir)

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    asr_proc = subprocess.Popen(
        asr_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    pronounce_proc = subprocess.Popen(
        pronounce_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )

    def stream_forwarder(pipe, prefix: str, dedicated_log, unified_log):
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                dedicated_log.write(line)
                unified_line = f"{prefix} {line}"
                unified_log.write(unified_line)
                sys.stdout.write(unified_line)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            pipe.close()

    t_asr = threading.Thread(
        target=stream_forwarder,
        args=(asr_proc.stdout, "[asr]", asr_log_f, serve_log_f),
        daemon=True,
    )
    t_pronounce = threading.Thread(
        target=stream_forwarder,
        args=(pronounce_proc.stdout, "[pronounce]", pronounce_log_f, serve_log_f),
        daemon=True,
    )
    t_asr.start()
    t_pronounce.start()

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
        asr_log_f.close()
        pronounce_log_f.close()
        serve_log_f.close()
        if getattr(args, "internal_worker", False):
            pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    logger.error("%s exited unexpectedly with code %d", name, ret)
                    for sibling_name, sibling_proc in procs:
                        if sibling_proc is not proc:
                            terminate_process(sibling_proc, sibling_name)
                    asr_log_f.close()
                    pronounce_log_f.close()
                    serve_log_f.close()
                    if getattr(args, "internal_worker", False):
                        pid_file.unlink(missing_ok=True)
                    return ret
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)
        return 0


def main(args: Optional[list[str]] = None) -> int:
    """Entry point dispatching CLI actions."""
    cli_args = parse_args(args)
    pid_file = Path(cli_args.pid_file).resolve() if cli_args.pid_file else get_default_pid_file()
    log_dir = Path(cli_args.log_dir).resolve() if cli_args.log_dir else get_default_log_dir()

    if cli_args.stop:
        return handle_stop(pid_file)

    if cli_args.status:
        return handle_status(pid_file, cli_args.host, cli_args.asr_port, cli_args.pronounce_port, log_dir)

    if cli_args.logs:
        return handle_logs(log_dir / "serve.log", follow=True)

    if cli_args.detach and not cli_args.internal_worker:
        return spawn_detached(cli_args)

    return run_combined(cli_args)


if __name__ == "__main__":
    sys.exit(main())
