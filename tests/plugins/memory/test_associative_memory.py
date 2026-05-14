"""Tests: plugins/memory/associative_memory.py — Hopfield-inspired content-addressable pattern matching."""

from __future__ import annotations

import pytest

from plugins.memory.associative_memory import (
    ASSOCIATIVE_MEMORY_REGISTRY,
    AssociativeMemory,
    Pattern,
)


@pytest.fixture
def mem():
    return AssociativeMemory(capacity=10)


@pytest.fixture
def mem_with_patterns():
    m = AssociativeMemory(capacity=10)
    m.store(Pattern("p1", [1.0, 0.0, 0.0]))
    m.store(Pattern("p2", [0.0, 1.0, 0.0]))
    m.store(Pattern("p3", [0.0, 0.0, 1.0]))
    return m


class TestPattern:
    def test_pattern_creation(self):
        p = Pattern("key1", [1.0, 2.0, 3.0])
        assert p.key == "key1"
        assert p.features == [1.0, 2.0, 3.0]

    def test_pattern_frozen(self):
        p = Pattern("k", [1.0])
        with pytest.raises((AttributeError, TypeError)):
            p.key = "other"  # type: ignore[misc]

    def test_pattern_empty_features(self):
        p = Pattern("empty", [])
        assert p.features == []


class TestAssociativeMemoryStore:
    def test_store_and_recall(self, mem):
        p = Pattern("key1", [1.0, 0.5, -0.5])
        mem.store(p)
        result = mem.recall(p.features, top_k=1)
        assert len(result) == 1
        assert result[0].key == "key1"

    def test_store_overwrites_same_key(self, mem):
        mem.store(Pattern("key1", [1.0, 0.0]))
        mem.store(Pattern("key1", [0.0, 1.0]))
        result = mem.recall([0.0, 1.0], top_k=1)
        assert result[0].key == "key1"

    def test_store_capacity_exceeded(self, mem):
        for i in range(10):
            mem.store(Pattern(f"key{i}", [float(i)]))
        with pytest.raises(ValueError, match="capacity"):
            mem.store(Pattern("extra", [99.0]))

    def test_store_overwrite_does_not_count_toward_capacity(self, mem):
        """Overwriting an existing key replaces features without consuming capacity.

        Source code: if key not in store AND len >= capacity → raise.
        Overwrite (key already exists) bypasses the capacity check.
        New key addition when full still raises.
        """
        for i in range(10):
            mem.store(Pattern(f"key{i}", [float(i)]))
        assert len(mem) == 10
        # Overwrite key0 (already exists) — bypasses capacity check
        mem.store(Pattern("key0", [999.0]))
        assert len(mem) == 10  # still 10 unique keys
        # Attempting to add an 11th NEW key when full raises
        with pytest.raises(ValueError, match="capacity"):
            mem.store(Pattern("key10", [10.0]))
        # key0 was successfully overwritten
        result = mem.recall([999.0], top_k=1)
        assert result[0].key == "key0"


class TestAssociativeMemoryRecall:
    def test_recall_top_k(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=2)
        assert len(result) == 2
        assert result[0].key == "p1"  # exact match first

    def test_recall_empty_store(self, mem):
        result = mem.recall([1.0, 0.0], top_k=3)
        assert result == []

    def test_recall_returns_most_similar(self, mem_with_patterns):
        # Query is between p1 and p2
        result = mem_with_patterns.recall([0.7, 0.7, 0.0], top_k=1)
        # Should return p1 or p2 (both are similar to [0.7, 0.7, 0.0])
        assert result[0].key in ("p1", "p2")

    def test_recall_k_greater_than_store(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=100)
        assert len(result) == 3  # only 3 stored

    def test_recall_empty_features(self, mem_with_patterns):
        result = mem_with_patterns.recall([], top_k=2)
        # Empty query — cosine of zero vectors is 0, should still return k items
        assert len(result) == 2


class TestAssociativeMemoryForget:
    def test_forget_existing(self, mem):
        mem.store(Pattern("key1", [1.0]))
        assert mem.forget("key1") is True
        assert len(mem) == 0

    def test_forget_nonexistent(self, mem):
        assert mem.forget("ghost") is False

    def test_forget_and_recall(self, mem):
        mem.store(Pattern("p1", [1.0, 0.0]))
        mem.store(Pattern("p2", [0.0, 1.0]))
        mem.forget("p1")
        result = mem.recall([1.0, 0.0], top_k=1)
        assert result[0].key == "p2"


class TestAssociativeMemorySimilarity:
    def test_similarity_positive(self):
        sim = AssociativeMemory.similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-9

    def test_similarity_perpendicular(self):
        sim = AssociativeMemory.similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-9

    def test_similarity_negative(self):
        sim = AssociativeMemory.similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-9

    def test_similarity_zero_vectors(self):
        sim = AssociativeMemory.similarity([], [])
        assert sim == 0.0

    def test_similarity_mismatched_length(self):
        """zip truncates to shorter vector; similarity computed on truncated prefix."""
        # [1.0, 0.5] vs [1.0] → zip gives [(1.0, 1.0)] → dot=1.0, |a|=sqrt(1.25), |b|=1.0
        sim = AssociativeMemory.similarity([1.0, 0.5], [1.0])
        assert 0.8 < sim < 1.0  # partial match on first dimension only


class TestAssociativeMemoryLen:
    def test_len_empty(self, mem):
        assert len(mem) == 0

    def test_len_after_stores(self, mem):
        mem.store(Pattern("a", [1.0]))
        mem.store(Pattern("b", [1.0]))
        assert len(mem) == 2


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in ASSOCIATIVE_MEMORY_REGISTRY
        assert ASSOCIATIVE_MEMORY_REGISTRY["default"] is AssociativeMemory

    def test_registry_instantiation(self):
        inst = ASSOCIATIVE_MEMORY_REGISTRY["default"]()
        assert isinstance(inst, AssociativeMemory)
