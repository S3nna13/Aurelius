"""Tests: plugins/memory/decaying_store.py — Time-decaying memory store for recency-weighted retrieval."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from plugins.memory.decaying_store import (
    DECAYING_MEMORY,
    DecayingMemoryItem,
    DecayingMemoryStore,
)


@pytest.fixture
def store():
    return DecayingMemoryStore(ttl_seconds=10.0)


class TestDecayingMemoryItem:
    def test_item_defaults_timestamp(self):
        item = DecayingMemoryItem(key="k", value="v")
        assert item.key == "k"
        assert item.value == "v"
        assert item.timestamp > 0

    def test_item_explicit_timestamp(self):
        item = DecayingMemoryItem(key="k", value="v", timestamp=1234.0)
        assert item.timestamp == 1234.0


class TestDecayingMemoryStore:
    def test_put_and_get(self, store):
        store.put("key1", "value1")
        assert store.get("key1") == "value1"

    def test_get_missing_key(self, store):
        assert store.get("ghost") is None

    def test_put_overwrites(self, store):
        store.put("key1", "v1")
        store.put("key1", "v2")
        assert store.get("key1") == "v2"

    def test_get_expired_returns_none(self, store):
        store.put("key1", "value1")
        with patch.object(time, "monotonic", return_value=store._items["key1"].timestamp + 15.0):
            assert store.get("key1") is None

    def test_get_expired_removes_item(self, store):
        store.put("key1", "value1")
        with patch.object(time, "monotonic", return_value=store._items["key1"].timestamp + 15.0):
            result = store.get("key1")
        assert result is None
        assert "key1" not in store._items

    def test_get_weighted_basic(self, store):
        store.put("key1", "value1")
        with patch.object(time, "monotonic", return_value=store._items["key1"].timestamp + 5.0):
            result = store.get_weighted("key1")
        assert result is not None
        assert result[0] == "value1"
        assert 0.0 <= result[1] <= 1.0

    def test_get_weighted_expired(self, store):
        store.put("key1", "value1")
        with patch.object(time, "monotonic", return_value=store._items["key1"].timestamp + 15.0):
            assert store.get_weighted("key1") is None

    def test_size_counts_alive_only(self, store):
        store.put("a", "x")
        store.put("b", "y")
        assert store.size() == 2
        with patch.object(time, "monotonic", return_value=store._items["a"].timestamp + 15.0):
            store.get("a")
        assert store.size() == 1

    def test_clear(self, store):
        store.put("a", "x")
        store.put("b", "y")
        store.clear()
        assert store.size() == 0

    def test_prune_removes_expired(self, store):
        """Prune removes ALL entries whose age exceeds ttl_seconds."""
        store.put("a", "x")
        store.put("b", "y")
        ts_a = store._items["a"].timestamp
        # Both inserted at same time; advance past ttl for both
        with patch.object(time, "monotonic", return_value=ts_a + 15.0):
            removed = store.prune()
        assert removed == 2, f"expected 2 expired, got {removed}"
        assert "a" not in store._items
        assert "b" not in store._items

    def test_prune_none_expired(self, store):
        store.put("a", "x")
        removed = store.prune()
        assert removed == 0

    def test_prune_empty(self, store):
        assert store.prune() == 0


class TestSingleton:
    def test_singleton_exists(self):
        assert isinstance(DECAYING_MEMORY, DecayingMemoryStore)

    def test_singleton_is_shared_instance(self):
        from plugins.memory.decaying_store import DECAYING_MEMORY as dm

        assert dm is DECAYING_MEMORY
