"""Tests for src.memory.working_memory."""

from __future__ import annotations

import time

import pytest

from src.memory.working_memory import WorkingMemory, WorkingMemorySlot, WORKING_MEMORY


# ---------------------------------------------------------------------------
# WorkingMemorySlot fields
# ---------------------------------------------------------------------------


def test_slot_key_stored():
    slot = WorkingMemorySlot(key="foo", value=42)
    assert slot.key == "foo"


def test_slot_value_stored():
    slot = WorkingMemorySlot(key="foo", value={"nested": True})
    assert slot.value == {"nested": True}


def test_slot_created_at_is_float():
    slot = WorkingMemorySlot(key="x", value=1)
    assert isinstance(slot.created_at, float)


def test_slot_default_ttl():
    slot = WorkingMemorySlot(key="x", value=1)
    assert slot.ttl_seconds == 60.0


def test_slot_custom_ttl():
    slot = WorkingMemorySlot(key="x", value=1, ttl_seconds=5.0)
    assert slot.ttl_seconds == 5.0


def test_slot_created_at_near_now():
    before = time.monotonic()
    slot = WorkingMemorySlot(key="x", value=1)
    after = time.monotonic()
    assert before <= slot.created_at <= after


# ---------------------------------------------------------------------------
# WorkingMemory.set + get round-trip
# ---------------------------------------------------------------------------


def test_set_get_roundtrip_string():
    wm = WorkingMemory()
    wm.set("greeting", "hello")
    assert wm.get("greeting") == "hello"


def test_set_get_roundtrip_int():
    wm = WorkingMemory()
    wm.set("count", 42)
    assert wm.get("count") == 42


def test_set_get_roundtrip_dict():
    wm = WorkingMemory()
    data = {"a": 1, "b": [1, 2, 3]}
    wm.set("data", data)
    assert wm.get("data") == data


def test_get_missing_key_returns_none():
    wm = WorkingMemory()
    assert wm.get("nonexistent") is None


def test_set_overwrites_existing_key():
    wm = WorkingMemory()
    wm.set("key", "original")
    wm.set("key", "updated")
    assert wm.get("key") == "updated"


def test_set_overwrites_does_not_grow_len():
    wm = WorkingMemory(capacity=4)
    wm.set("key", "v1")
    assert len(wm) == 1
    wm.set("key", "v2")
    assert len(wm) == 1


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_get_expired_slot_returns_none():
    wm = WorkingMemory()
    wm.set("temp", "gone", ttl_seconds=0.0)
    # Slot has ttl_seconds=0, so any positive elapsed time means expired
    time.sleep(0.01)
    assert wm.get("temp") is None


def test_is_expired_fresh_slot_false():
    wm = WorkingMemory()
    slot = WorkingMemorySlot(key="x", value=1, ttl_seconds=60.0)
    assert wm.is_expired(slot) is False


def test_is_expired_old_slot_true():
    wm = WorkingMemory()
    slot = WorkingMemorySlot(key="x", value=1, ttl_seconds=0.0)
    time.sleep(0.01)
    assert wm.is_expired(slot) is True


def test_evict_expired_removes_expired():
    wm = WorkingMemory()
    wm.set("fresh", "here", ttl_seconds=60.0)
    wm.set("expired", "gone", ttl_seconds=0.0)
    # Allow time to pass so "expired" slot is past TTL, then call evict directly
    time.sleep(0.01)
    count = wm.evict_expired()
    assert count == 1
    assert wm.get("fresh") == "here"


def test_evict_expired_returns_count():
    wm = WorkingMemory()
    # Bypass set() to avoid internal evict_expired calls; inject slots directly
    wm._slots["c"] = WorkingMemorySlot(key="c", value=3, ttl_seconds=60.0)
    wm._slots["a"] = WorkingMemorySlot(key="a", value=1, ttl_seconds=0.0)
    wm._slots["b"] = WorkingMemorySlot(key="b", value=2, ttl_seconds=0.0)
    time.sleep(0.01)
    count = wm.evict_expired()
    assert count == 2


def test_evict_expired_on_empty_returns_zero():
    wm = WorkingMemory()
    assert wm.evict_expired() == 0


# ---------------------------------------------------------------------------
# keys()
# ---------------------------------------------------------------------------


def test_keys_non_expired_only():
    wm = WorkingMemory()
    wm.set("alive", "yes", ttl_seconds=60.0)
    wm.set("dead", "no", ttl_seconds=0.0)
    time.sleep(0.01)
    keys = wm.keys()
    assert "alive" in keys
    assert "dead" not in keys


def test_keys_empty_store():
    wm = WorkingMemory()
    assert wm.keys() == []


def test_keys_returns_list():
    wm = WorkingMemory()
    wm.set("x", 1)
    assert isinstance(wm.keys(), list)


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_zero_initially():
    wm = WorkingMemory()
    assert len(wm) == 0


def test_len_counts_non_expired():
    wm = WorkingMemory()
    wm.set("a", 1, ttl_seconds=60.0)
    wm.set("b", 2, ttl_seconds=0.0)
    time.sleep(0.01)
    assert len(wm) == 1


# ---------------------------------------------------------------------------
# Capacity eviction
# ---------------------------------------------------------------------------


def test_capacity_eviction_at_limit():
    wm = WorkingMemory(capacity=16)
    for i in range(17):
        wm.set(f"key{i}", i)
    assert len(wm) == 16


def test_capacity_eviction_removes_oldest():
    wm = WorkingMemory(capacity=3)
    wm.set("first", 1)
    time.sleep(0.01)
    wm.set("second", 2)
    time.sleep(0.01)
    wm.set("third", 3)
    time.sleep(0.01)
    wm.set("fourth", 4)  # Should evict "first"
    assert wm.get("first") is None
    assert wm.get("second") is not None


def test_capacity_eviction_newest_preserved():
    wm = WorkingMemory(capacity=3)
    for i in range(5):
        wm.set(f"k{i}", i)
    # Last 3 should survive (capacity-based, not TTL)
    assert len(wm) == 3


# ---------------------------------------------------------------------------
# WORKING_MEMORY singleton
# ---------------------------------------------------------------------------


def test_working_memory_singleton_exists():
    assert isinstance(WORKING_MEMORY, WorkingMemory)


def test_working_memory_singleton_is_working_memory():
    assert type(WORKING_MEMORY).__name__ == "WorkingMemory"
