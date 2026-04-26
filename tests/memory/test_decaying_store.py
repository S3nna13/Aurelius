"""Tests for decaying memory store."""

from __future__ import annotations

import time

from src.memory.decaying_store import DecayingMemoryStore


class TestDecayingMemoryStore:
    def test_put_and_get(self):
        store = DecayingMemoryStore(ttl_seconds=3600)
        store.put("k1", "v1")
        assert store.get("k1") == "v1"

    def test_get_expired_returns_none(self):
        store = DecayingMemoryStore(ttl_seconds=0)
        store.put("k", "v")
        time.sleep(0.01)
        assert store.get("k") is None

    def test_get_weighted(self):
        store = DecayingMemoryStore(ttl_seconds=3600)
        store.put("k", "v")
        val, weight = store.get_weighted("k")
        assert val == "v"
        assert weight > 0.9

    def test_size_excludes_expired(self):
        store = DecayingMemoryStore(ttl_seconds=0)
        store.put("k", "v")
        time.sleep(0.01)
        assert store.size() == 0

    def test_prune_removes_expired(self):
        store = DecayingMemoryStore(ttl_seconds=0)
        store.put("k", "v")
        time.sleep(0.01)
        count = store.prune()
        assert count == 1
        assert store.size() == 0

    def test_clear(self):
        store = DecayingMemoryStore()
        store.put("k", "v")
        store.clear()
        assert store.size() == 0
