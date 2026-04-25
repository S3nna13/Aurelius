"""Tests for CLI config."""
from __future__ import annotations

import json
import tempfile

import pytest

from src.cli.config import CLIAppConfig


class TestCLIAppConfig:
    def test_default_values(self):
        c = CLIAppConfig()
        assert c.log_level == "INFO"
        assert c.theme == "dark"

    def test_save_and_load(self):
        c = CLIAppConfig(log_level="DEBUG", theme="light")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        c.save(path)
        loaded = CLIAppConfig.load(path)
        assert loaded.log_level == "DEBUG"
        assert loaded.theme == "light"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("AURELIUS_LOG_LEVEL", "WARN")
        monkeypatch.setenv("AURELIUS_THEME", "solarized")
        c = CLIAppConfig.from_env()
        assert c.log_level == "WARN"
        assert c.theme == "solarized"