"""Tests for honeytoken manager."""
from __future__ import annotations

import pytest

from src.security.honeytoken_manager import HoneytokenManager


class TestHoneytokenManager:
    def test_generates_token(self):
        hm = HoneytokenManager()
        token = hm.generate("db://users")
        assert token.token.startswith("hk_")

    def test_check_unknown_returns_none(self):
        hm = HoneytokenManager()
        assert hm.check("nonexistent") is None

    def test_check_marks_accessed(self):
        hm = HoneytokenManager()
        t = hm.generate("file://config")
        hm.check(t.token, source="unknown_ip")
        assert len(hm.alerts()) == 1
        assert hm.alerts()[0].access_source == "unknown_ip"