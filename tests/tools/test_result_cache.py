"""Tests for result cache."""

from __future__ import annotations

import time

from src.tools.result_cache import ResultCache


class TestResultCache:
    def test_set_and_get(self):
        rc = ResultCache()
        rc.set("k1", {"data": 42})
        assert rc.get("k1") == {"data": 42}

    def test_expired_returns_none(self):
        rc = ResultCache(default_ttl=0)
        rc.set("k", "v")
        time.sleep(0.01)
        assert rc.get("k") is None

    def test_invalidate_all(self):
        rc = ResultCache()
        rc.set("a", 1)
        rc.set("b", 2)
        assert rc.invalidate() == 2
        assert rc.size() == 0

    def test_invalidate_prefix(self):
        rc = ResultCache()
        rc.set("user:1", "alice")
        rc.set("user:2", "bob")
        rc.set("config", "x")
        assert rc.invalidate("user:") == 2
        assert rc.size() == 1

    def test_cleanup(self):
        rc = ResultCache(default_ttl=0)
        rc.set("x", 1)
        time.sleep(0.01)
        assert rc.cleanup() == 1
