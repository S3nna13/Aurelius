"""Tests for src.memory.layered_memory."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from src.memory.layered_memory import (
    DEFAULT_LAYERED_MEMORY,
    LAYERED_MEMORY_REGISTRY,
    LayeredMemory,
    LayeredMemoryEntry,
    LayeredMemoryError,
    MemoryLayer,
)

# ---------------------------------------------------------------------------
# LayeredMemoryEntry
# ---------------------------------------------------------------------------


def test_entry_id_is_8_char_hex():
    entry = LayeredMemoryEntry(content="hello")
    assert re.fullmatch(r"[0-9a-f]{8}", entry.entry_id)


def test_entry_timestamp_is_utc_datetime():
    entry = LayeredMemoryEntry(content="hello")
    assert isinstance(entry.timestamp, datetime)
    assert entry.timestamp.tzinfo is not None


def test_entry_defaults():
    entry = LayeredMemoryEntry()
    assert entry.content == ""
    assert entry.layer == ""
    assert entry.access_count == 0
    assert entry.importance_score == 0.5


# ---------------------------------------------------------------------------
# MemoryLayer
# ---------------------------------------------------------------------------


def test_memory_layer_fields():
    layer = MemoryLayer(level=2, name="L2", ttl_seconds=30, max_entries=500)
    assert layer.level == 2
    assert layer.name == "L2"
    assert layer.ttl_seconds == 30
    assert layer.max_entries == 500
    assert layer.entries == []


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_store_string_creates_entry():
    mem = LayeredMemory()
    entry = mem.store("hello world", "L4 Session Archive")
    assert isinstance(entry, LayeredMemoryEntry)
    assert entry.content == "hello world"
    assert entry.layer == "L4 Session Archive"


def test_store_entry_object():
    mem = LayeredMemory()
    entry = LayeredMemoryEntry(content="obj")
    result = mem.store(entry, "L1 Insight Index")
    assert result is entry
    assert result.layer == "L1 Insight Index"


def test_store_increases_len():
    mem = LayeredMemory()
    assert len(mem) == 0
    mem.store("a", "L4 Session Archive")
    assert len(mem) == 1


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


def test_retrieve_exact_match():
    mem = LayeredMemory()
    mem.store("The quick brown fox", "L4 Session Archive")
    results = mem.retrieve("quick")
    assert len(results) == 1
    assert results[0].content == "The quick brown fox"


def test_retrieve_case_insensitive():
    mem = LayeredMemory()
    mem.store("Hello World", "L4 Session Archive")
    results = mem.retrieve("hello")
    assert len(results) == 1


def test_retrieve_layer_isolation():
    mem = LayeredMemory()
    mem.store("alpha", "L2 Global Facts")
    mem.store("alpha", "L4 Session Archive")
    results = mem.retrieve("alpha", layer_name="L2 Global Facts")
    assert len(results) == 1
    assert results[0].layer == "L2 Global Facts"


def test_retrieve_increments_access_count():
    mem = LayeredMemory()
    mem.store("target", "L4 Session Archive")
    mem.retrieve("target")
    results = mem.retrieve("target")
    assert results[0].access_count >= 2


def test_retrieve_no_match():
    mem = LayeredMemory()
    assert mem.retrieve("missing") == []


# ---------------------------------------------------------------------------
# evict_expired
# ---------------------------------------------------------------------------


def test_evict_expired_removes_old_entries():
    mem = LayeredMemory()
    old_entry = LayeredMemoryEntry(
        content="old",
        layer="L4 Session Archive",
        timestamp=datetime.now(UTC) - timedelta(days=2),
    )
    mem.store(old_entry, "L4 Session Archive")
    mem.store("fresh", "L4 Session Archive")
    count = mem.evict_expired()
    assert count == 1
    assert len(mem) == 1


def test_evict_expired_respects_layer_ttl():
    mem = LayeredMemory()
    old_l1 = LayeredMemoryEntry(
        content="old l1",
        layer="L1 Insight Index",
        timestamp=datetime.now(UTC) - timedelta(days=3),
    )
    old_l4 = LayeredMemoryEntry(
        content="old l4",
        layer="L4 Session Archive",
        timestamp=datetime.now(UTC) - timedelta(days=2),
    )
    mem.store(old_l1, "L1 Insight Index")
    mem.store(old_l4, "L4 Session Archive")
    count = mem.evict_expired()
    assert count == 1  # only L4 expired
    assert len(mem.dump_layer("L1 Insight Index")) == 1


def test_evict_expired_l0_never_evicted():
    mem = LayeredMemory()
    old_l0 = LayeredMemoryEntry(
        content="bootstrap",
        layer="L0 Meta Rules",
        timestamp=datetime.now(UTC) - timedelta(days=365),
    )
    mem.store(old_l0, "L0 Meta Rules")
    assert mem.evict_expired() == 0
    assert len(mem.dump_layer("L0 Meta Rules")) == 1


# ---------------------------------------------------------------------------
# max_entries eviction
# ---------------------------------------------------------------------------


def test_max_entries_eviction():
    mem = LayeredMemory()
    layer = mem._layers["L3 Task Skills"]
    layer.max_entries = 3
    for i in range(5):
        mem.store(f"skill {i}", "L3 Task Skills")
    assert len(mem.dump_layer("L3 Task Skills")) == 3


def test_max_entries_eviction_uses_importance_recency():
    mem = LayeredMemory()
    layer = mem._layers["L1 Insight Index"]
    layer.max_entries = 2
    old_low = LayeredMemoryEntry(
        content="old low",
        timestamp=datetime.now(UTC) - timedelta(hours=10),
        importance_score=0.1,
    )
    new_high = LayeredMemoryEntry(
        content="new high",
        timestamp=datetime.now(UTC),
        importance_score=0.9,
    )
    mid = LayeredMemoryEntry(
        content="mid",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
        importance_score=0.5,
    )
    mem.store(old_low, "L1 Insight Index")
    mem.store(new_high, "L1 Insight Index")
    mem.store(mid, "L1 Insight Index")
    dumped = mem.dump_layer("L1 Insight Index")
    contents = {e.content for e in dumped}
    assert "old low" not in contents
    assert "new high" in contents


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_moves_up_layer():
    mem = LayeredMemory()
    entry = mem.store("promote me", "L2 Global Facts", access_count=3, importance_score=0.5)
    assert entry.layer == "L2 Global Facts"
    ok = mem.promote(entry.entry_id)
    assert ok is True
    assert entry.layer == "L1 Insight Index"
    assert any(e.entry_id == entry.entry_id for e in mem.dump_layer("L1 Insight Index"))


def test_promote_fails_at_l0():
    mem = LayeredMemory()
    entry = mem.store("top", "L0 Meta Rules", access_count=5, importance_score=0.9)
    assert mem.promote(entry.entry_id) is False


def test_promote_requires_criteria():
    mem = LayeredMemory()
    entry = mem.store("weak", "L3 Task Skills", access_count=0, importance_score=0.1)
    assert mem.promote(entry.entry_id) is False


def test_promote_unknown_entry_raises():
    mem = LayeredMemory()
    with pytest.raises(LayeredMemoryError):
        mem.promote("deadbeef")


# ---------------------------------------------------------------------------
# dump_layer
# ---------------------------------------------------------------------------


def test_dump_layer_returns_all():
    mem = LayeredMemory()
    mem.store("a", "L4 Session Archive")
    mem.store("b", "L4 Session Archive")
    assert len(mem.dump_layer("L4 Session Archive")) == 2


def test_dump_layer_unknown_raises():
    mem = LayeredMemory()
    with pytest.raises(LayeredMemoryError):
        mem.dump_layer("L99 Fake Layer")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_ranks_by_relevance():
    mem = LayeredMemory()
    mem.store("alpha beta gamma", "L4 Session Archive", importance_score=0.9)
    mem.store("alpha", "L4 Session Archive", importance_score=0.1)
    results = mem.search("alpha", top_k=1)
    assert len(results) == 1
    assert "beta gamma" in results[0].content


def test_search_respects_top_k():
    mem = LayeredMemory()
    for i in range(10):
        mem.store(f"common word {i}", "L4 Session Archive")
    results = mem.search("common", top_k=3)
    assert len(results) == 3


def test_search_empty_query_matches_all():
    mem = LayeredMemory()
    mem.store("something", "L4 Session Archive")
    results = mem.search("", top_k=5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_store_unknown_layer_raises():
    mem = LayeredMemory()
    with pytest.raises(LayeredMemoryError):
        mem.store("x", "L99 Fake Layer")


def test_retrieve_unknown_layer_raises():
    mem = LayeredMemory()
    with pytest.raises(LayeredMemoryError):
        mem.retrieve("x", layer_name="L99 Fake Layer")


# ---------------------------------------------------------------------------
# Singleton / registry
# ---------------------------------------------------------------------------


def test_default_layered_memory_singleton_exists():
    assert isinstance(DEFAULT_LAYERED_MEMORY, LayeredMemory)


def test_registry_contains_default():
    assert "default" in LAYERED_MEMORY_REGISTRY
    assert LAYERED_MEMORY_REGISTRY["default"] is DEFAULT_LAYERED_MEMORY
