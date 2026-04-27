"""Integration tests for the Aurelius CLI (src/cli/main.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI_MODULE = "src.cli.main"
PROJECT_ROOT = Path(__file__).parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI with given arguments and return the result."""
    return subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


class TestCLIVersion:
    def test_version_flag(self) -> None:
        result = run_cli("--version")
        assert result.returncode == 0
        assert "aurelius" in result.stdout.lower() or "0.1" in result.stdout

    def test_version_short_flag(self) -> None:
        result = run_cli("-V")
        assert result.returncode == 0


class TestCLIConfig:
    def test_config_list(self) -> None:
        result = run_cli("config", "list")
        assert result.returncode == 0
        assert "d_model" in result.stdout
        assert "n_layers" in result.stdout

    def test_config_get(self) -> None:
        result = run_cli("config", "get", "d_model")
        assert result.returncode == 0
        assert "2048" in result.stdout

    def test_config_get_unknown_key(self) -> None:
        result = run_cli("config", "get", "nonexistent_key")
        assert result.returncode == 1
        assert "Unknown" in result.stdout

    def test_config_get_missing_key_arg(self) -> None:
        result = run_cli("config", "get")
        assert "Usage" in result.stdout

    def test_config_path(self) -> None:
        result = run_cli("config", "path")
        assert result.returncode == 0
        assert "configs" in result.stdout

    def test_config_set_readonly(self) -> None:
        result = run_cli("config", "set", "d_model", "4096")
        assert "read-only" in result.stdout


class TestCLIHealth:
    def test_health(self) -> None:
        result = run_cli("health")
        assert result.returncode == 0
        assert "Model Config" in result.stdout or "PyTorch" in result.stdout or "Health" in result.stdout

    def test_health_verbose(self) -> None:
        result = run_cli("health", "--verbose")
        assert result.returncode == 0

    def test_health_short_flag(self) -> None:
        result = run_cli("health", "-v")
        assert result.returncode == 0


class TestCLIHelp:
    def test_help(self) -> None:
        result = run_cli("--help")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_chat_help(self) -> None:
        result = run_cli("chat", "--help")
        assert result.returncode == 0

    def test_serve_help(self) -> None:
        result = run_cli("serve", "--help")
        assert result.returncode == 0

    def test_train_help(self) -> None:
        result = run_cli("train", "--help")
        assert result.returncode == 0

    def test_eval_help(self) -> None:
        result = run_cli("eval", "--help")
        assert result.returncode == 0


class TestCLIError:
    def test_unknown_command(self) -> None:
        result = run_cli("nonexistent_command")
        assert result.returncode != 0

    def test_train_no_config(self) -> None:
        result = run_cli("train")
        assert result.returncode == 0 or result.returncode == 1
