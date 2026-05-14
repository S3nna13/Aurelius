"""Tests: plugins/memory/layered_memory.py — 5-tier memory hierarchy with TTL and promotion."""

from __future__ import annotations

import time

import pytest

from plugins.memory.layered_memory import (
    DEFAULT_LAYERED_MEMORY,
    LAYERED_MEMORY_REGISTRY,
    LayeredMemory,
    LayeredMemoryEntry,
    LayeredMemoryError,
    MemoryLayer,
)


class TestLayeredMemoryEntry:
    """LayeredMemoryEntry dataclass."""

    def test_entry_defaults(self):
        entry = LayeredMemoryEntry(content="test content", layer="L0 Meta Rules")
        assert entry.content == "test content"
        assert entry.layer == "L0 Meta Rules"
        assert entry.access_count == 0
        assert entry.importance_score == 0.5
        assert len(entry.entry_id) == 8


class TestMemoryLayer:
    """MemoryLayer dataclass."""

    def test_layer_defaults(self):
        layer = MemoryLayer(level=0, name="Test", ttl_seconds=3600, max_entries=100)
        assert layer.level == 0
        assert layer.name == "Test"
        assert layer.ttl_seconds == 3600
        assert layer.max_entries == 100
        assert layer.entries == []


class TestLayeredMemory:
    """5-layer memory hierarchy."""

    def test_default_layers_exist(self):
        mem = LayeredMemory()
        assert "L0 Meta Rules" in mem._layers
        assert "L1 Insight Index" in mem._layers
        assert "L2 Global Facts" in mem._layers
        assert "L3 Task Skills" in mem._layers
        assert "L4 Session Archive" in mem._layers

    def test_custom_layers(self):
        custom = MemoryLayer(level=0, name="Custom", ttl_seconds=60, max_entries=10)
        mem = LayeredMemory(layers=[custom])
        assert "Custom" in mem._layers
        assert mem._layers["Custom"].max_entries == 10

    def test_store_string_entry(self):
        mem = LayeredMemory()
        entry = mem.store("test content", "L0 Meta Rules")
        assert entry.content == "test content"
        assert entry.layer == "L0 Meta Rules"
        assert len(mem) == 1

    def test_store_entry_object(self):
        mem = LayeredMemory()
        entry_obj = LayeredMemoryEntry(content="object entry", layer="L0 Meta Rules")
        entry = mem.store(entry_obj, "L0 Meta Rules")
        assert entry.content == "object entry"

    def test_store_unknown_layer_raises(self):
        mem = LayeredMemory()
        with pytest.raises(LayeredMemoryError, match="Unknown layer"):
            mem.store("content", "NonExistentLayer")

    def test_capacity_eviction(self):
        custom = MemoryLayer(level=0, name="Small", ttl_seconds=None, max_entries=2)
        mem = LayeredMemory(layers=[custom])
        for i in range(5):
            mem.store(f"Entry {i}", "Small")
        layer = mem._layers["Small"]
        assert len(layer.entries) == 2

    def test_retrieve_single_layer(self):
        mem = LayeredMemory()
        mem.store("hello world", "L0 Meta Rules")
        mem.store("goodbye world", "L1 Insight Index")
        results = mem.retrieve("world", layer_name="L0 Meta Rules")
        assert len(results) == 1
        assert results[0].content == "hello world"

    def test_retrieve_all_layers(self):
        mem = LayeredMemory()
        mem.store("apple", "L0 Meta Rules")
        mem.store("apple", "L1 Insight Index")
        results = mem.retrieve("apple")
        assert len(results) == 2

    def test_retrieve_increments_access_count(self):
        mem = LayeredMemory()
        entry = mem.store("test content", "L0 Meta Rules")
        assert entry.access_count == 0
        mem.retrieve("test", layer_name="L0 Meta Rules")
        updated = mem._layers["L0 Meta Rules"].entries[0]
        assert updated.access_count == 1

    def test_search(self):
        mem = LayeredMemory()
        mem.store("Python is a programming language", "L2 Global Facts")
        mem.store("Rust is fast", "L2 Global Facts")
        mem.store("Python decorators are powerful", "L3 Task Skills")
        results = mem.search("Python", top_k=5)
        assert len(results) >= 2
        for r in results:
            assert "python" in r.content.lower()

    def test_search_empty_query(self):
        mem = LayeredMemory()
        mem.store("something", "L0 Meta Rules")
        results = mem.search("")
        assert len(results) == 1  # empty query matches everything

    def test_search_no_matches(self):
        mem = LayeredMemory()
        mem.store("apple banana", "L0 Meta Rules")
        results = mem.search("zebra")
        assert results == []

    def test_promote_success(self):
        mem = LayeredMemory()
        entry = mem.store("high value entry", "L1 Insight Index", importance_score=0.8)
        # Access 3 times to meet promotion criteria
        for _ in range(3):
            mem.retrieve("high value", layer_name="L1 Insight Index")
        entry.access_count = 3
        result = mem.promote(entry.entry_id)
        assert result is True
        # Entry should now be in L0
        assert len(mem._layers["L0 Meta Rules"].entries) == 1

    def test_promote_fails_insufficient_criteria(self):
        mem = LayeredMemory()
        entry = mem.store("low value entry", "L1 Insight Index", importance_score=0.3)
        entry.access_count = 1
        result = mem.promote(entry.entry_id)
        assert result is False

    def test_promote_top_layer_fails(self):
        mem = LayeredMemory()
        entry = mem.store("meta rule", "L0 Meta Rules")
        entry.access_count = 5
        entry.importance_score = 0.9
        result = mem.promote(entry.entry_id)
        assert result is False

    def test_promote_not_found_raises(self):
        mem = LayeredMemory()
        with pytest.raises(LayeredMemoryError, match="Entry not found"):
            mem.promote("nonexistent-id")

    def test_evict_expired(self):
        custom = MemoryLayer(level=0, name="TTLTest", ttl_seconds=1, max_entries=100)
        mem = LayeredMemory(layers=[custom])
        mem.store("will expire", "TTLTest")
        time.sleep(1.1)
        removed = mem.evict_expired()
        assert removed >= 1
        assert len(mem.dump_layer("TTLTest")) == 0

    def test_dump_layer(self):
        mem = LayeredMemory()
        mem.store("one", "L0 Meta Rules")
        mem.store("two", "L0 Meta Rules")
        entries = mem.dump_layer("L0 Meta Rules")
        assert len(entries) == 2

    def test_dump_layer_unknown_raises(self):
        mem = LayeredMemory()
        with pytest.raises(LayeredMemoryError, match="Unknown layer"):
            mem.dump_layer("FakeLayer")

    def test_len(self):
        mem = LayeredMemory()
        assert len(mem) == 0
        mem.store("first", "L0 Meta Rules")
        assert len(mem) == 1
        mem.store("second", "L1 Insight Index")
        assert len(mem) == 2


class TestLayeredMemoryRegistry:
    """Module-level registry and singleton."""

    def test_default_singleton_exists(self):
        assert DEFAULT_LAYERED_MEMORY is not None
        assert isinstance(DEFAULT_LAYERED_MEMORY, LayeredMemory)

    def test_registry_contains_default(self):
        assert "default" in LAYERED_MEMORY_REGISTRY
        assert LAYERED_MEMORY_REGISTRY["default"] is DEFAULT_LAYERED_MEMORY
