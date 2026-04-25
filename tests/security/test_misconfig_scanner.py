"""Tests for misconfig scanner."""
from __future__ import annotations

import pytest

from src.security.misconfig_scanner import MisconfigScanner, MisconfigCheck


class TestMisconfigScanner:
    def test_passing_check(self):
        ms = MisconfigScanner()
        ms.add_check(MisconfigCheck("debug_mode", lambda: (True, "ok")))
        results = ms.run_all()
        assert results[0]["passed"] is True

    def test_failing_check(self):
        ms = MisconfigScanner()
        ms.add_check(MisconfigCheck("ssl_disabled", lambda: (False, "SSL not enabled")))
        results = ms.run_all()
        assert results[0]["passed"] is False

    def test_check_exception(self):
        ms = MisconfigScanner()
        def broken():
            raise RuntimeError("boom")
        ms.add_check(MisconfigCheck("broken", broken))
        results = ms.run_all()
        assert results[0]["passed"] is False
        assert "boom" in results[0]["message"]

    def test_failures_only(self):
        ms = MisconfigScanner()
        ms.add_check(MisconfigCheck("pass", lambda: (True, "")))
        ms.add_check(MisconfigCheck("fail", lambda: (False, "fail")))
        assert len(ms.failures()) == 1