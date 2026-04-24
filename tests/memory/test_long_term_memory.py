"""Tests for src.memory.long_term_memory — 25+ tests."""
from __future__ import annotations

import math
import time

import pytest

from src.memory.long_term_memory import (
    LTMEntry,
    LongTermMemory,
    LONG_TERM_MEMORY,
    _MAX_KEY_LEN,
    _MAX_VALUE_STR_LEN,
    _MAX_CAPACITY,
    _MAX_TAGS,
)


# ---------------------------------------------------------------------------
# LTMEntry construction and basic validation
# ---------------------------------------------------------------------------


def test_ltm_entry_basic_fields():
    entry = LTMEntry(key="k", value="v", importance=0.5)
    assert entry.key == "k"
    assert entry.value == "v"
    assert entry.importance == 0.5


def test_ltm_entry_importance_zero_valid():
    e = LTMEntry(key="k", value=1, importance=0.0)
    assert e.importance == 0.0


def test_ltm_entry_importance_one_valid():
    e = LTMEntry(key="k", value=1, importance=1.0)
    assert e.importance == 1.0


def test_ltm_entry_importance_below_zero_raises():
    with pytest.raises(ValueError, match="importance"):
        LTMEntry(key="k", value=1, importance=-0.01)


def test_ltm_entry_importance_above_one_raises():
    with pytest.raises(ValueError, match="importance"):
        LTMEntry(key="k", value=1, importance=1.01)


def test_ltm_entry_too_many_tags_raises():
    tags = frozenset(str(i) for i in range(_MAX_TAGS + 1))
    with pytest.raises(ValueError, match="too many tags"):
        LTMEntry(key="k", value=1, importance=0.5, tags=tags)


def test_ltm_entry_max_tags_ok():
    tags = frozenset(str(i) for i in range(_MAX_TAGS))
    e = LTMEntry(key="k", value=1, importance=0.5, tags=tags)
    assert len(e.tags) == _MAX_TAGS


def test_ltm_entry_access_count_default_zero():
    e = LTMEntry(key="k", value=1, importance=0.5)
    assert e.access_count == 0


def test_ltm_entry_created_at_is_float():
    e = LTMEntry(key="k", value=1, importance=0.5)
    assert isinstance(e.created_at, float)


# ---------------------------------------------------------------------------
# decayed_importance
# ---------------------------------------------------------------------------


def test_decayed_importance_zero_elapsed():
    now = time.monotonic()
    e = LTMEntry(key="k", value=1, importance=0.8, created_at=now)
    # elapsed is ~0, decay ~ 1.0
    result = e.decayed_importance(now, decay_rate=0.01)
    assert abs(result - 0.8) < 1e-6


def test_decayed_importance_zero_importance_always_zero():
    now = time.monotonic()
    e = LTMEntry(key="k", value=1, importance=0.0, created_at=now - 7200)
    assert e.decayed_importance(now) == 0.0


def test_decayed_importance_higher_rate_decays_faster():
    now = time.monotonic()
    past = now - 3600  # 1 hour ago
    e = LTMEntry(key="k", value=1, importance=1.0, created_at=past)
    slow = e.decayed_importance(now, decay_rate=0.01)
    fast = e.decayed_importance(now, decay_rate=0.5)
    assert fast < slow


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_access_count_increases_score():
    now = time.monotonic()
    e = LTMEntry(key="k", value=1, importance=0.5, created_at=now, last_accessed=now)
    score_before = e.score(now)
    e.access_count = 10
    score_after = e.score(now)
    assert score_after > score_before


# ---------------------------------------------------------------------------
# LongTermMemory construction
# ---------------------------------------------------------------------------


def test_ltm_capacity_exceeds_max_raises():
    with pytest.raises(ValueError, match="capacity exceeds"):
        LongTermMemory(capacity=_MAX_CAPACITY + 1)


def test_ltm_capacity_zero_raises():
    with pytest.raises(ValueError, match="capacity must be"):
        LongTermMemory(capacity=0)


def test_ltm_capacity_negative_raises():
    with pytest.raises(ValueError, match="capacity must be"):
        LongTermMemory(capacity=-1)


# ---------------------------------------------------------------------------
# store + retrieve round-trip
# ---------------------------------------------------------------------------


def test_store_retrieve_basic():
    ltm = LongTermMemory()
    ltm.store("hello", "world", importance=0.6)
    entry = ltm.retrieve("hello")
    assert entry is not None
    assert entry.value == "world"


def test_retrieve_missing_key_returns_none():
    ltm = LongTermMemory()
    assert ltm.retrieve("nope") is None


def test_retrieve_increments_access_count():
    ltm = LongTermMemory()
    ltm.store("x", 1, importance=0.5)
    ltm.retrieve("x")
    ltm.retrieve("x")
    entry = ltm._store["x"]
    assert entry.access_count == 2


def test_retrieve_updates_last_accessed():
    ltm = LongTermMemory()
    ltm.store("x", 1, importance=0.5)
    before = ltm._store["x"].last_accessed
    time.sleep(0.01)
    ltm.retrieve("x")
    after = ltm._store["x"].last_accessed
    assert after >= before


# ---------------------------------------------------------------------------
# store validation
# ---------------------------------------------------------------------------


def test_store_key_too_long_raises():
    ltm = LongTermMemory()
    with pytest.raises(ValueError, match="key exceeds"):
        ltm.store("a" * (_MAX_KEY_LEN + 1), "v")


def test_store_string_value_too_long_raises():
    ltm = LongTermMemory()
    with pytest.raises(ValueError, match="string value exceeds"):
        ltm.store("k", "x" * (_MAX_VALUE_STR_LEN + 1))


def test_store_non_string_large_value_ok():
    ltm = LongTermMemory()
    # list with many elements — no size constraint on non-str values
    ltm.store("biglist", list(range(100_000)))
    assert ltm.retrieve("biglist") is not None


# ---------------------------------------------------------------------------
# Duplicate key overwrites without eviction
# ---------------------------------------------------------------------------


def test_duplicate_key_overwrites_no_eviction():
    ltm = LongTermMemory(capacity=2)
    ltm.store("a", 1, importance=0.5)
    ltm.store("b", 2, importance=0.5)
    assert len(ltm) == 2
    # Overwrite "a" — should NOT evict "b"
    ltm.store("a", 99, importance=0.9)
    assert len(ltm) == 2
    assert ltm.retrieve("a").value == 99
    assert ltm.retrieve("b") is not None


# ---------------------------------------------------------------------------
# Capacity eviction
# ---------------------------------------------------------------------------


def test_capacity_eviction_on_overflow():
    ltm = LongTermMemory(capacity=3)
    ltm.store("a", 1, importance=0.5)
    ltm.store("b", 2, importance=0.5)
    ltm.store("c", 3, importance=0.5)
    ltm.store("d", 4, importance=0.5)   # must evict one
    assert len(ltm) == 3


def test_capacity_one_second_store_evicts_first():
    ltm = LongTermMemory(capacity=1)
    ltm.store("first", "v1", importance=0.5)
    ltm.store("second", "v2", importance=0.5)
    assert len(ltm) == 1
    assert ltm.retrieve("second") is not None
    assert ltm.retrieve("first") is None


# ---------------------------------------------------------------------------
# forget()
# ---------------------------------------------------------------------------


def test_forget_existing_key_returns_true():
    ltm = LongTermMemory()
    ltm.store("x", 1)
    assert ltm.forget("x") is True


def test_forget_existing_key_removes_entry():
    ltm = LongTermMemory()
    ltm.store("x", 1)
    ltm.forget("x")
    assert ltm.retrieve("x") is None


def test_forget_missing_key_returns_false():
    ltm = LongTermMemory()
    assert ltm.forget("no_such_key") is False


# ---------------------------------------------------------------------------
# __len__ and __contains__
# ---------------------------------------------------------------------------


def test_len_empty():
    ltm = LongTermMemory()
    assert len(ltm) == 0


def test_len_after_stores():
    ltm = LongTermMemory()
    ltm.store("a", 1)
    ltm.store("b", 2)
    assert len(ltm) == 2


def test_contains_present():
    ltm = LongTermMemory()
    ltm.store("key", "val")
    assert "key" in ltm


def test_contains_absent():
    ltm = LongTermMemory()
    assert "missing" not in ltm


# ---------------------------------------------------------------------------
# top_k()
# ---------------------------------------------------------------------------


def test_top_k_returns_correct_count():
    ltm = LongTermMemory()
    for i in range(10):
        ltm.store(f"k{i}", i, importance=i / 10)
    result = ltm.top_k(5)
    assert len(result) == 5


def test_top_k_sorted_by_score_descending():
    ltm = LongTermMemory()
    for i in range(5):
        ltm.store(f"k{i}", i, importance=i / 10)
    now = time.monotonic()
    result = ltm.top_k(5)
    scores = [e.score(now, ltm.decay_rate) for e in result]
    assert scores == sorted(scores, reverse=True)


def test_top_k_zero_raises():
    ltm = LongTermMemory()
    ltm.store("x", 1)
    with pytest.raises(ValueError, match="k must be"):
        ltm.top_k(0)


def test_top_k_larger_than_store_returns_all():
    ltm = LongTermMemory()
    ltm.store("a", 1)
    ltm.store("b", 2)
    result = ltm.top_k(100)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# search_by_tags()
# ---------------------------------------------------------------------------


def test_search_by_tags_any_match():
    ltm = LongTermMemory()
    ltm.store("a", 1, tags={"cat", "dog"})
    ltm.store("b", 2, tags={"cat"})
    ltm.store("c", 3, tags={"bird"})
    results = ltm.search_by_tags({"cat"}, require_all=False)
    keys = {e.key for e in results}
    assert "a" in keys and "b" in keys and "c" not in keys


def test_search_by_tags_require_all():
    ltm = LongTermMemory()
    ltm.store("a", 1, tags={"cat", "dog"})
    ltm.store("b", 2, tags={"cat"})
    results = ltm.search_by_tags({"cat", "dog"}, require_all=True)
    keys = {e.key for e in results}
    assert "a" in keys and "b" not in keys


def test_search_by_tags_empty_tags_returns_nothing():
    ltm = LongTermMemory()
    ltm.store("a", 1, tags={"cat"})
    # empty tags set: no intersection ever
    results = ltm.search_by_tags(set(), require_all=False)
    assert results == []


def test_search_by_tags_require_all_empty_tags():
    ltm = LongTermMemory()
    ltm.store("a", 1, tags={"cat"})
    # empty set is subset of every frozenset → all entries match
    results = ltm.search_by_tags(set(), require_all=True)
    assert len(results) == 1


def test_search_by_tags_sorted_by_score():
    ltm = LongTermMemory()
    ltm.store("low", 1, importance=0.1, tags={"t"})
    ltm.store("high", 2, importance=0.9, tags={"t"})
    results = ltm.search_by_tags({"t"})
    assert results[0].key == "high"


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


def test_empty_key_is_valid():
    ltm = LongTermMemory()
    ltm.store("", "empty key value", importance=0.5)
    assert ltm.retrieve("") is not None


def test_key_with_control_chars_stored_as_is():
    ltm = LongTermMemory()
    key = "key\x00\x01\x1f"
    ltm.store(key, "ctrl", importance=0.5)
    assert ltm.retrieve(key) is not None


# ---------------------------------------------------------------------------
# LONG_TERM_MEMORY singleton
# ---------------------------------------------------------------------------


def test_long_term_memory_singleton_exists():
    assert isinstance(LONG_TERM_MEMORY, LongTermMemory)
