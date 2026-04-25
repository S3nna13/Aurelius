"""Tests for src.memory.episodic_memory."""

from __future__ import annotations

import re

import pytest

from src.memory.episodic_memory import EpisodicMemory, MemoryEntry


# ---------------------------------------------------------------------------
# MemoryEntry construction
# ---------------------------------------------------------------------------


def test_memory_entry_id_is_8_char_hex():
    entry = MemoryEntry(role="user", content="hello")
    assert re.fullmatch(r"[0-9a-f]{8}", entry.id), f"bad id: {entry.id!r}"


def test_memory_entry_two_ids_differ():
    e1 = MemoryEntry(role="user", content="a")
    e2 = MemoryEntry(role="user", content="b")
    assert e1.id != e2.id


def test_memory_entry_timestamp_is_iso8601():
    from datetime import datetime, timezone
    entry = MemoryEntry(role="user", content="ts test")
    # Should be parseable as ISO 8601 datetime
    dt = datetime.fromisoformat(entry.timestamp)
    assert dt.tzinfo is not None  # timezone-aware


def test_memory_entry_role_stored():
    entry = MemoryEntry(role="assistant", content="hi")
    assert entry.role == "assistant"


def test_memory_entry_content_stored():
    entry = MemoryEntry(role="user", content="my content")
    assert entry.content == "my content"


def test_memory_entry_default_importance():
    entry = MemoryEntry(role="user", content="x")
    assert entry.importance == 1.0


def test_memory_entry_custom_importance():
    entry = MemoryEntry(role="user", content="x", importance=0.3)
    assert entry.importance == 0.3


# ---------------------------------------------------------------------------
# EpisodicMemory.store
# ---------------------------------------------------------------------------


def test_store_returns_memory_entry():
    mem = EpisodicMemory()
    result = mem.store("user", "hello")
    assert isinstance(result, MemoryEntry)


def test_store_increases_len():
    mem = EpisodicMemory()
    assert len(mem) == 0
    mem.store("user", "a")
    assert len(mem) == 1
    mem.store("user", "b")
    assert len(mem) == 2


def test_store_preserves_role():
    mem = EpisodicMemory()
    entry = mem.store("assistant", "response")
    assert entry.role == "assistant"


def test_store_preserves_content():
    mem = EpisodicMemory()
    entry = mem.store("user", "question?")
    assert entry.content == "question?"


def test_store_preserves_importance():
    mem = EpisodicMemory()
    entry = mem.store("user", "x", importance=0.7)
    assert entry.importance == 0.7


# ---------------------------------------------------------------------------
# retrieve_recent
# ---------------------------------------------------------------------------


def test_retrieve_recent_empty():
    mem = EpisodicMemory()
    assert mem.retrieve_recent(5) == []


def test_retrieve_recent_returns_n():
    mem = EpisodicMemory()
    for i in range(10):
        mem.store("user", f"msg {i}")
    result = mem.retrieve_recent(3)
    assert len(result) == 3


def test_retrieve_recent_returns_last_n():
    mem = EpisodicMemory()
    entries = [mem.store("user", f"msg {i}") for i in range(10)]
    result = mem.retrieve_recent(3)
    assert [e.id for e in result] == [e.id for e in entries[-3:]]


def test_retrieve_recent_all_when_n_exceeds_len():
    mem = EpisodicMemory()
    for i in range(3):
        mem.store("user", f"msg {i}")
    result = mem.retrieve_recent(10)
    assert len(result) == 3


def test_retrieve_recent_order_is_chronological():
    mem = EpisodicMemory()
    e1 = mem.store("user", "first")
    e2 = mem.store("user", "second")
    e3 = mem.store("user", "third")
    result = mem.retrieve_recent(3)
    assert result[0].id == e1.id
    assert result[1].id == e2.id
    assert result[2].id == e3.id


# ---------------------------------------------------------------------------
# retrieve_by_importance
# ---------------------------------------------------------------------------


def test_retrieve_by_importance_filters_below_threshold():
    mem = EpisodicMemory()
    mem.store("user", "low", importance=0.2)
    mem.store("user", "high", importance=0.8)
    result = mem.retrieve_by_importance(0.5)
    assert len(result) == 1
    assert result[0].content == "high"


def test_retrieve_by_importance_includes_equal_threshold():
    mem = EpisodicMemory()
    entry = mem.store("user", "edge", importance=0.5)
    result = mem.retrieve_by_importance(0.5)
    assert any(e.id == entry.id for e in result)


def test_retrieve_by_importance_sorted_descending():
    mem = EpisodicMemory()
    mem.store("user", "a", importance=0.6)
    mem.store("user", "b", importance=0.9)
    mem.store("user", "c", importance=0.7)
    result = mem.retrieve_by_importance(0.0)
    importances = [e.importance for e in result]
    assert importances == sorted(importances, reverse=True)


def test_retrieve_by_importance_empty_store():
    mem = EpisodicMemory()
    assert mem.retrieve_by_importance(0.5) == []


def test_retrieve_by_importance_none_qualify():
    mem = EpisodicMemory()
    mem.store("user", "x", importance=0.1)
    result = mem.retrieve_by_importance(0.9)
    assert result == []


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_exact_match():
    mem = EpisodicMemory()
    mem.store("user", "The quick brown fox")
    result = mem.search("quick")
    assert len(result) == 1


def test_search_case_insensitive():
    mem = EpisodicMemory()
    mem.store("user", "Hello World")
    result = mem.search("hello")
    assert len(result) == 1


def test_search_case_insensitive_query_upper():
    mem = EpisodicMemory()
    mem.store("user", "lowercase content")
    result = mem.search("LOWERCASE")
    assert len(result) == 1


def test_search_no_match_returns_empty():
    mem = EpisodicMemory()
    mem.store("user", "apple")
    result = mem.search("orange")
    assert result == []


def test_search_returns_multiple_matches():
    mem = EpisodicMemory()
    mem.store("user", "cat sat on mat")
    mem.store("user", "cat ran away")
    mem.store("user", "dog slept")
    result = mem.search("cat")
    assert len(result) == 2


def test_search_empty_store():
    mem = EpisodicMemory()
    assert mem.search("anything") == []


def test_search_returns_memory_entry_objects():
    mem = EpisodicMemory()
    mem.store("user", "test content")
    result = mem.search("test")
    assert all(isinstance(e, MemoryEntry) for e in result)


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


def test_forget_returns_true_for_known_id():
    mem = EpisodicMemory()
    entry = mem.store("user", "forget me")
    assert mem.forget(entry.id) is True


def test_forget_returns_false_for_unknown_id():
    mem = EpisodicMemory()
    assert mem.forget("nonexistent") is False


def test_forget_reduces_len():
    mem = EpisodicMemory()
    entry = mem.store("user", "gone")
    assert len(mem) == 1
    mem.forget(entry.id)
    assert len(mem) == 0


def test_forget_entry_not_retrievable_after():
    mem = EpisodicMemory()
    entry = mem.store("user", "unique content xyz")
    mem.forget(entry.id)
    result = mem.search("unique content xyz")
    assert result == []


def test_forget_only_removes_target():
    mem = EpisodicMemory()
    e1 = mem.store("user", "keep me")
    e2 = mem.store("user", "remove me")
    mem.forget(e2.id)
    assert len(mem) == 1
    result = mem.search("keep me")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_returns_count():
    mem = EpisodicMemory()
    for i in range(5):
        mem.store("user", f"msg {i}")
    count = mem.clear()
    assert count == 5


def test_clear_empties_memory():
    mem = EpisodicMemory()
    mem.store("user", "a")
    mem.store("user", "b")
    mem.clear()
    assert len(mem) == 0


def test_clear_on_empty_returns_zero():
    mem = EpisodicMemory()
    assert mem.clear() == 0


# ---------------------------------------------------------------------------
# max_entries eviction
# ---------------------------------------------------------------------------


def test_max_entries_evicts_oldest():
    mem = EpisodicMemory(max_entries=3)
    entries = [mem.store("user", f"msg {i}") for i in range(5)]
    assert len(mem) == 3


def test_max_entries_retains_newest():
    mem = EpisodicMemory(max_entries=3)
    entries = [mem.store("user", f"msg {i}") for i in range(5)]
    recent = mem.retrieve_recent(3)
    # Should have the last 3 entries
    assert [e.id for e in recent] == [e.id for e in entries[-3:]]


def test_max_entries_one():
    mem = EpisodicMemory(max_entries=1)
    e1 = mem.store("user", "first")
    e2 = mem.store("user", "second")
    assert len(mem) == 1
    result = mem.retrieve_recent(1)
    assert result[0].id == e2.id
