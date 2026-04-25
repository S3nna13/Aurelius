"""Tests for priority memory store."""
from __future__ import annotations

import pytest

from src.memory.priority_store import PriorityMemoryStore, MemoryItem, Priority


class TestPriorityMemoryStore:
    def test_put_and_get(self):
        store = PriorityMemoryStore(max_size=10)
        store.put(MemoryItem("k1", "v1"))
        assert store.get("k1") == "v1"

    def test_get_missing_returns_none(self):
        store = PriorityMemoryStore()
        assert store.get("nonexistent") is None

    def test_evicts_lowest_priority(self):
        store = PriorityMemoryStore(max_size=2)
        store.put(MemoryItem("k1", "v1", Priority.LOW))
        store.put(MemoryItem("k2", "v2", Priority.HIGH))
        store.put(MemoryItem("k3", "v3", Priority.MEDIUM))
        assert store.get("k1") is None  # evicted
        assert store.get("k2") == "v2"
        assert store.get("k3") == "v3"

    def test_size(self):
        store = PriorityMemoryStore()
        assert store.size() == 0
        store.put(MemoryItem("k1", "v1"))
        assert store.size() == 1

    def test_clear(self):
        store = PriorityMemoryStore()
        store.put(MemoryItem("k1", "v1"))
        store.clear()
        assert store.size() == 0