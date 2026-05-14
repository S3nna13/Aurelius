"""Tests: plugins/memory/priority_store.py — Priority-based memory eviction policy."""

from __future__ import annotations

import pytest

from plugins.memory.priority_store import (
    PRIORITY_MEMORY,
    MemoryItem,
    Priority,
    PriorityMemoryStore,
)


@pytest.fixture
def store():
    return PriorityMemoryStore(max_size=3)


class TestPriority:
    def test_priority_ordering(self):
        """Enum auto() assigns ascending integers. LOW < MEDIUM < HIGH < CRITICAL."""
        assert Priority.LOW.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.CRITICAL.value


class TestMemoryItem:
    def test_item_defaults(self):
        item = MemoryItem(key="k", value="v")
        assert item.key == "k"
        assert item.value == "v"
        assert item.priority is Priority.LOW
        assert item.access_count == 0

    def test_item_explicit_priority(self):
        item = MemoryItem(key="k", value="v", priority=Priority.CRITICAL)
        assert item.priority is Priority.CRITICAL


class TestPriorityMemoryStore:
    def test_put_and_get(self, store):
        item = MemoryItem(key="a", value="x", priority=Priority.HIGH)
        store.put(item)
        assert store.get("a") == "x"

    def test_get_missing_key(self, store):
        assert store.get("ghost") is None

    def test_access_count_increments(self, store):
        store.put(MemoryItem("a", "x"))
        store.get("a")
        store.get("a")
        assert store._items["a"].access_count == 2

    def test_eviction_lowest_priority(self, store):
        """When full, lowest-priority item should be evicted."""
        store.put(MemoryItem("a", "x", priority=Priority.LOW))
        store.put(MemoryItem("b", "y", priority=Priority.MEDIUM))
        store.put(MemoryItem("c", "z", priority=Priority.HIGH))
        # Now full — adding HIGH should evict nothing
        store.put(MemoryItem("d", "w", priority=Priority.CRITICAL))
        assert store.size() == 3
        # LOW was evicted
        assert store.get("a") is None

    def test_eviction_tiebreaker_access_count(self, store):
        """Same priority — lower access_count evicted first; stable sort breaks ties.

        Eviction triggers when len(items) >= max_size (on the NEXT put).
        With max_size=3: after putting a,b,c store is full (3 items).
        When d is put, _evict_lowest_priority() fires.
        At that point: a=(2,1), b=(2,0), c=(2,0). Sorted ascending: b first
        (stable sort: b inserted before c, both have same (priority, access_count)).
        So 'b' is evicted, 'a' and 'c' survive.
        """
        store.put(MemoryItem("a", "x", priority=Priority.MEDIUM))
        store.put(MemoryItem("b", "y", priority=Priority.MEDIUM))
        store.get("a")  # access_count: a=1, b=0
        # Store now has a,b,c (3 items, full). Putting d triggers eviction.
        store.put(MemoryItem("c", "z", priority=Priority.MEDIUM))  # 3rd item, no eviction yet
        store.put(MemoryItem("d", "w", priority=Priority.MEDIUM))  # 4th item → eviction
        assert store.get("b") is None  # b evicted (lowest access_count among ties)
        assert store.get("a") == "x"  # a survives (higher access_count)
        assert store.get("c") == "z"  # c survives (inserted before d, same tie)

    def test_overwrite_does_not_count_as_eviction(self, store):
        store.put(MemoryItem("a", "v1", priority=Priority.LOW))
        store.put(MemoryItem("a", "v2", priority=Priority.HIGH))
        assert store.size() == 1
        assert store.get("a") == "v2"

    def test_size(self, store):
        store.put(MemoryItem("a", "x"))
        store.put(MemoryItem("b", "y"))
        assert store.size() == 2

    def test_clear(self, store):
        store.put(MemoryItem("a", "x"))
        store.put(MemoryItem("b", "y"))
        store.clear()
        assert store.size() == 0


class TestSingleton:
    def test_priority_memory_singleton(self):
        assert isinstance(PRIORITY_MEMORY, PriorityMemoryStore)
