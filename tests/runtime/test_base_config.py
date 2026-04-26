"""Tests for base config."""

from __future__ import annotations

from src.runtime.base_config import BaseConfig


class TestBaseConfig:
    def test_get_returns_default(self):
        bc = BaseConfig()
        assert bc.get("nonexistent", "fallback") == "fallback"

    def test_coerce_bool(self):
        bc = BaseConfig()
        assert bc._coerce("true") is True
        assert bc._coerce("false") is False

    def test_coerce_int(self):
        bc = BaseConfig()
        assert bc._coerce("42") == 42

    def test_to_dict(self):
        bc = BaseConfig()
        d = bc.to_dict()
        assert isinstance(d, dict)
        assert "_prefix" not in d
