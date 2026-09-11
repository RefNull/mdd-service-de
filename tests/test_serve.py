"""Tests for the combined local deployment runner (serve.py)."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest

from serve import build_commands, parse_args, run_combined, terminate_process


def test_parse_args_defaults():
    """Verify default CLI arguments."""
    args = parse_args([])
    assert args.host == "0.0.0.0"
    assert args.asr_port == 8001
    assert args.pronounce_port == 8002
    assert args.model == "Qwen3-ASR-0.6B"
    assert args.device == "auto"


def test_parse_args_custom():
    """Verify custom CLI argument parsing."""
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--asr-port",
            "9001",
            "--pronounce-port",
            "9002",
            "--model",
            "Qwen3-ASR-1.7B",
            "--device",
            "cuda",
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.asr_port == 9001
    assert args.pronounce_port == 9002
    assert args.model == "Qwen3-ASR-1.7B"
    assert args.device == "cuda"


def test_parse_args_port_collision():
    """Verify that using identical ports for ASR and Pronounce triggers an error."""
    with pytest.raises(SystemExit):
        parse_args(["--asr-port", "8080", "--pronounce-port", "8080"])


def test_parse_args_env_model(monkeypatch):
    """Verify that MDD_ASR_MODEL env var sets the default model."""
    monkeypatch.setenv("MDD_ASR_MODEL", "Qwen3-ASR-1.7B")
    args = parse_args([])
    assert args.model == "Qwen3-ASR-1.7B"


def test_build_commands():
    """Verify construction of subprocess execution commands."""
    args = parse_args(["--host", "127.0.0.1", "--asr-port", "8010", "--pronounce-port", "8020", "--model", "custom-model", "--device", "cpu"])
    asr_cmd, pronounce_cmd = build_commands(args, python_exe="/opt/python")

    assert asr_cmd == [
        "/opt/python",
        "asr_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8010",
        "--device",
        "cpu",
        "--model",
        "custom-model",
    ]
    assert pronounce_cmd == [
        "/opt/python",
        "pronounce_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8020",
        "--device",
        "cpu",
    ]


def test_terminate_process_already_exited():
    """Verify terminate_process does nothing if process has already stopped."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0

    terminate_process(mock_proc, "TestService")
    mock_proc.terminate.assert_not_called()
    mock_proc.kill.assert_not_called()


def test_terminate_process_graceful():
    """Verify terminate_process calls terminate() and waits."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345

    terminate_process(mock_proc, "TestService", timeout=1.0)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=1.0)
    mock_proc.kill.assert_not_called()


def test_terminate_process_timeout_fallback_to_kill():
    """Verify terminate_process sends kill() if terminate() times out."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=1.0), None]

    terminate_process(mock_proc, "TestService", timeout=1.0)
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


@patch("serve.subprocess.Popen")
@patch("serve.signal.signal")
def test_run_combined_process_exit_supervision(mock_signal, mock_popen):
    """Verify run_combined terminates the sibling process if one exits unexpectedly."""
    proc1 = MagicMock()
    proc1.pid = 101
    proc1.poll.return_value = None

    proc2 = MagicMock()
    proc2.pid = 102
    # Simulate proc2 exiting unexpectedly with error code 1
    proc2.poll.return_value = 1

    mock_popen.side_effect = [proc1, proc2]

    args = parse_args([])
    exit_code = run_combined(args, poll_interval=0.01)

    assert exit_code == 1
    # Check that sibling proc1 was terminated
    proc1.terminate.assert_called_once()
