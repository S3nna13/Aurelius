"""Tests for KV store."""

from __future__ import annotations

import time

from src.tools.kv_store import KVStore


class TestKVStore:
    def test_put_and_get(self):
        kv = KVStore()
        kv.put("key1", "value1")
        assert kv.get("key1") == "value1"

    def test_get_missing(self):
        kv = KVStore()
        assert kv.get("nonexistent") is None

    def test_ttl_expiry(self):
        kv = KVStore()
        kv.put("temp", "data", ttl=0.001)
        time.sleep(0.01)
        assert kv.get("temp") is None

    def test_delete(self):
        kv = KVStore()
        kv.put("a", 1)
        kv.delete("a")
        assert kv.exists("a") is False

    def test_cleanup(self):
        kv = KVStore()
        kv.put("e1", "v", ttl=0.001)
        kv.put("e2", "v", ttl=0.001)
        time.sleep(0.01)
        n = kv.cleanup()
        assert n == 2

    def test_size(self):
        kv = KVStore()
        kv.put("a", 1)
        kv.put("b", 2)
        assert kv.size() == 2
