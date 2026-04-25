"""
Tests for src/memory/associative_memory.py

Coverage: Pattern, AssociativeMemory, ASSOCIATIVE_MEMORY_REGISTRY.
At least 28 test functions.
"""

import math

import pytest

from src.memory.associative_memory import (
    ASSOCIATIVE_MEMORY_REGISTRY,
    AssociativeMemory,
    Pattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem():
    return AssociativeMemory(capacity=10)


@pytest.fixture()
def mem_with_patterns():
    m = AssociativeMemory(capacity=10)
    m.store(Pattern("p1", [1.0, 0.0, 0.0]))
    m.store(Pattern("p2", [0.0, 1.0, 0.0]))
    m.store(Pattern("p3", [0.0, 0.0, 1.0]))
    return m


# ---------------------------------------------------------------------------
# Pattern dataclass
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AssociativeMemory: store
# ---------------------------------------------------------------------------

class TestStore:
    def test_store_one_pattern(self, mem):
        mem.store(Pattern("a", [1.0, 0.0]))
        assert len(mem) == 1

    def test_store_multiple_patterns(self, mem):
        mem.store(Pattern("a", [1.0, 0.0]))
        mem.store(Pattern("b", [0.0, 1.0]))
        assert len(mem) == 2

    def test_store_overwrites_same_key(self, mem):
        mem.store(Pattern("a", [1.0, 0.0]))
        mem.store(Pattern("a", [0.0, 1.0]))
        assert len(mem) == 1
        recalled = mem.recall([0.0, 1.0], top_k=1)
        assert recalled[0].features == [0.0, 1.0]

    def test_store_capacity_exceeded_raises(self):
        m = AssociativeMemory(capacity=2)
        m.store(Pattern("a", [1.0]))
        m.store(Pattern("b", [2.0]))
        with pytest.raises(ValueError, match="capacity"):
            m.store(Pattern("c", [3.0]))

    def test_store_overwrite_does_not_consume_capacity(self):
        m = AssociativeMemory(capacity=1)
        m.store(Pattern("a", [1.0]))
        # Overwriting same key must not raise
        m.store(Pattern("a", [2.0]))
        assert len(m) == 1


# ---------------------------------------------------------------------------
# AssociativeMemory: recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_recall_empty_returns_empty(self, mem):
        assert mem.recall([1.0, 0.0]) == []

    def test_recall_top_k_1_exact(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=1)
        assert len(result) == 1
        assert result[0].key == "p1"

    def test_recall_top_k_2(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=2)
        assert len(result) == 2
        assert result[0].key == "p1"

    def test_recall_top_k_exceeds_stored(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=100)
        assert len(result) == 3

    def test_recall_returns_correct_order(self, mem_with_patterns):
        # Query pointing toward p2
        result = mem_with_patterns.recall([0.0, 1.0, 0.0], top_k=3)
        assert result[0].key == "p2"

    def test_recall_default_top_k_is_1(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0])
        assert len(result) == 1

    def test_recall_single_stored_pattern(self, mem):
        mem.store(Pattern("only", [1.0, 1.0]))
        result = mem.recall([1.0, 1.0], top_k=1)
        assert result[0].key == "only"

    def test_recall_returns_pattern_objects(self, mem_with_patterns):
        result = mem_with_patterns.recall([1.0, 0.0, 0.0], top_k=1)
        assert isinstance(result[0], Pattern)


# ---------------------------------------------------------------------------
# AssociativeMemory: forget
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_existing_returns_true(self, mem):
        mem.store(Pattern("x", [1.0]))
        assert mem.forget("x") is True

    def test_forget_existing_decrements_len(self, mem):
        mem.store(Pattern("x", [1.0]))
        mem.forget("x")
        assert len(mem) == 0

    def test_forget_missing_returns_false(self, mem):
        assert mem.forget("nonexistent") is False

    def test_forget_then_store_new(self, mem):
        m = AssociativeMemory(capacity=1)
        m.store(Pattern("a", [1.0]))
        m.forget("a")
        m.store(Pattern("b", [2.0]))  # should not raise
        assert len(m) == 1

    def test_forget_does_not_affect_other_patterns(self, mem_with_patterns):
        mem_with_patterns.forget("p1")
        assert len(mem_with_patterns) == 2
        assert mem_with_patterns.recall([0.0, 1.0, 0.0], top_k=1)[0].key == "p2"


# ---------------------------------------------------------------------------
# AssociativeMemory: __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_len_empty(self, mem):
        assert len(mem) == 0

    def test_len_after_stores(self, mem):
        for i in range(5):
            mem.store(Pattern(str(i), [float(i)]))
        assert len(mem) == 5

    def test_len_after_forget(self, mem):
        mem.store(Pattern("a", [1.0]))
        mem.store(Pattern("b", [2.0]))
        mem.forget("a")
        assert len(mem) == 1


# ---------------------------------------------------------------------------
# AssociativeMemory: similarity (static method)
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_similarity_identical_vectors_is_1(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(AssociativeMemory.similarity(v, v), 1.0, rel_tol=1e-6)

    def test_similarity_orthogonal_is_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(AssociativeMemory.similarity(a, b), 0.0, abs_tol=1e-9)

    def test_similarity_zero_vector_no_crash(self):
        zero = [0.0, 0.0, 0.0]
        other = [1.0, 2.0, 3.0]
        result = AssociativeMemory.similarity(zero, other)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_similarity_both_zero_no_crash(self):
        zero = [0.0, 0.0]
        result = AssociativeMemory.similarity(zero, zero)
        assert isinstance(result, float)

    def test_similarity_parallel_vectors_is_1(self):
        a = [2.0, 4.0, 6.0]
        b = [1.0, 2.0, 3.0]
        assert math.isclose(AssociativeMemory.similarity(a, b), 1.0, rel_tol=1e-6)

    def test_similarity_antiparallel_is_minus_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(AssociativeMemory.similarity(a, b), -1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(ASSOCIATIVE_MEMORY_REGISTRY, dict)

    def test_registry_has_default(self):
        assert "default" in ASSOCIATIVE_MEMORY_REGISTRY

    def test_registry_default_is_associative_memory(self):
        assert ASSOCIATIVE_MEMORY_REGISTRY["default"] is AssociativeMemory

    def test_registry_default_instantiates(self):
        cls = ASSOCIATIVE_MEMORY_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, AssociativeMemory)
