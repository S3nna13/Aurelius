"""Tests for src/multiagent/agent_memory.py."""

from __future__ import annotations

import time

import pytest

from src.multiagent.agent_memory import (
    AGENT_MEMORY_REGISTRY,
    AgentMemory,
    MemoryEntry,
    MemoryType,
)


def _mem() -> AgentMemory:
    return AgentMemory(agent_id="a1")


def test_registry_has_default():
    assert "default" in AGENT_MEMORY_REGISTRY
    assert AGENT_MEMORY_REGISTRY["default"] is AgentMemory


def test_memory_type_values():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"
    assert MemoryType.WORKING.value == "working"


def test_init_sets_agent_id_and_empty():
    m = AgentMemory("alice")
    assert m.agent_id == "alice"
    assert m.all_entries() == []


def test_store_returns_entry():
    m = _mem()
    e = m.store("hello", MemoryType.EPISODIC)
    assert isinstance(e, MemoryEntry)
    assert e.content == "hello"
    assert e.memory_type == MemoryType.EPISODIC


def test_store_default_importance():
    m = _mem()
    e = m.store("x", MemoryType.SEMANTIC)
    assert e.importance == 0.5


def test_store_custom_importance_and_tags():
    m = _mem()
    e = m.store("x", MemoryType.SEMANTIC, importance=0.9, tags=["a", "b"])
    assert e.importance == 0.9
    assert e.tags == ["a", "b"]


def test_store_ttl_propagates():
    m = _mem()
    e = m.store("x", MemoryType.WORKING, ttl_s=5.0)
    assert e.ttl_s == 5.0


def test_store_auto_entry_id_len_8():
    m = _mem()
    e = m.store("x", MemoryType.SEMANTIC)
    assert len(e.entry_id) == 8


def test_entry_ids_unique():
    m = _mem()
    ids = {m.store(f"c{i}", MemoryType.SEMANTIC).entry_id for i in range(20)}
    assert len(ids) == 20


def test_recall_keyword_in_content():
    m = _mem()
    m.store("the cat sat", MemoryType.EPISODIC)
    m.store("dog barked", MemoryType.EPISODIC)
    results = m.recall("cat")
    assert len(results) == 1
    assert "cat" in results[0].content


def test_recall_keyword_in_tags():
    m = _mem()
    m.store("unrelated content", MemoryType.SEMANTIC, tags=["physics"])
    results = m.recall("physics")
    assert len(results) == 1


def test_recall_case_insensitive():
    m = _mem()
    m.store("Hello World", MemoryType.SEMANTIC)
    assert len(m.recall("hello")) == 1
    assert len(m.recall("WORLD")) == 1


def test_recall_sorted_by_importance_desc():
    m = _mem()
    m.store("cat one", MemoryType.SEMANTIC, importance=0.2)
    m.store("cat two", MemoryType.SEMANTIC, importance=0.9)
    m.store("cat three", MemoryType.SEMANTIC, importance=0.5)
    out = m.recall("cat")
    assert [e.importance for e in out] == [0.9, 0.5, 0.2]


def test_recall_filters_by_memory_type():
    m = _mem()
    m.store("cat episodic", MemoryType.EPISODIC)
    m.store("cat semantic", MemoryType.SEMANTIC)
    out = m.recall("cat", memory_type=MemoryType.SEMANTIC)
    assert len(out) == 1
    assert out[0].memory_type == MemoryType.SEMANTIC


def test_recall_top_k():
    m = _mem()
    for i in range(10):
        m.store(f"cat {i}", MemoryType.SEMANTIC, importance=i / 10.0)
    out = m.recall("cat", top_k=3)
    assert len(out) == 3


def test_recall_no_match_returns_empty():
    m = _mem()
    m.store("hello", MemoryType.SEMANTIC)
    assert m.recall("xyz") == []


def test_recall_multiword_query_any_match():
    m = _mem()
    m.store("apple pie", MemoryType.SEMANTIC)
    m.store("banana split", MemoryType.SEMANTIC)
    out = m.recall("apple banana")
    assert len(out) == 2


def test_forget_existing_returns_true():
    m = _mem()
    e = m.store("x", MemoryType.SEMANTIC)
    assert m.forget(e.entry_id) is True
    assert m.all_entries() == []


def test_forget_missing_returns_false():
    m = _mem()
    assert m.forget("nope1234") is False


def test_expire_old_removes_expired():
    m = _mem()
    m.store("old", MemoryType.WORKING, ttl_s=1.0)
    future = time.monotonic() + 100.0
    removed = m.expire_old(current_time=future)
    assert removed == 1
    assert m.all_entries() == []


def test_expire_old_keeps_no_ttl():
    m = _mem()
    m.store("permanent", MemoryType.SEMANTIC)
    removed = m.expire_old(current_time=time.monotonic() + 1e9)
    assert removed == 0
    assert len(m.all_entries()) == 1


def test_expire_old_keeps_unexpired():
    m = _mem()
    m.store("fresh", MemoryType.WORKING, ttl_s=1000.0)
    removed = m.expire_old()
    assert removed == 0


def test_consolidate_removes_low_importance():
    m = _mem()
    m.store("a", MemoryType.SEMANTIC, importance=0.1)
    m.store("b", MemoryType.SEMANTIC, importance=0.5)
    m.store("c", MemoryType.SEMANTIC, importance=0.9)
    removed = m.consolidate(min_importance=0.3)
    assert removed == 1
    assert len(m.all_entries()) == 2


def test_consolidate_threshold_zero_keeps_all():
    m = _mem()
    m.store("a", MemoryType.SEMANTIC, importance=0.0)
    removed = m.consolidate(min_importance=0.0)
    assert removed == 0


def test_all_entries_filter_by_type():
    m = _mem()
    m.store("a", MemoryType.EPISODIC)
    m.store("b", MemoryType.SEMANTIC)
    m.store("c", MemoryType.SEMANTIC)
    assert len(m.all_entries(MemoryType.SEMANTIC)) == 2
    assert len(m.all_entries(MemoryType.EPISODIC)) == 1


def test_stats_counts_and_average():
    m = _mem()
    m.store("a", MemoryType.EPISODIC, importance=0.2)
    m.store("b", MemoryType.SEMANTIC, importance=0.8)
    s = m.stats()
    assert s["total"] == 2
    assert s["count_by_type"]["episodic"] == 1
    assert s["count_by_type"]["semantic"] == 1
    assert abs(s["avg_importance"] - 0.5) < 1e-9


def test_stats_empty():
    m = _mem()
    s = m.stats()
    assert s["total"] == 0
    assert s["avg_importance"] == 0.0
    assert all(v == 0 for v in s["count_by_type"].values())


def test_max_entries_eviction():
    m = AgentMemory("a", max_entries=3)
    m.store("low", MemoryType.SEMANTIC, importance=0.1)
    m.store("mid", MemoryType.SEMANTIC, importance=0.5)
    m.store("hi", MemoryType.SEMANTIC, importance=0.9)
    m.store("new", MemoryType.SEMANTIC, importance=0.7)
    contents = {e.content for e in m.all_entries()}
    assert "low" not in contents
    assert len(m.all_entries()) == 3


def test_memory_entry_frozen():
    e = MemoryEntry(content="x", memory_type=MemoryType.SEMANTIC)
    with pytest.raises(Exception):
        e.content = "y"  # type: ignore[misc]
