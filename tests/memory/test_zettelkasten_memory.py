"""Tests for ZettelkastenMemory (A-MEM pattern — arXiv 2502.12110)."""

from __future__ import annotations

import pytest

from src.memory.zettelkasten_memory import ZettelkastenMemory, ZettelNote


@pytest.fixture()
def mem() -> ZettelkastenMemory:
    return ZettelkastenMemory(max_notes=10, similarity_threshold=0.5)


class TestAddAndSize:
    def test_add_returns_note(self, mem):
        note = mem.add("hello world", tags=["greeting"])
        assert isinstance(note, ZettelNote)

    def test_size_increments(self, mem):
        assert mem.size() == 0
        mem.add("first")
        assert mem.size() == 1
        mem.add("second")
        assert mem.size() == 2

    def test_note_fields_populated(self, mem):
        note = mem.add("content", tags=["a", "b"], importance=0.8)
        assert note.content == "content"
        assert note.tags == ["a", "b"]
        assert note.importance == 0.8
        assert note.note_id != ""
        assert note.created_at > 0

    def test_importance_clamped_high(self, mem):
        note = mem.add("x", importance=1.5)
        assert note.importance == 1.0

    def test_importance_clamped_low(self, mem):
        note = mem.add("x", importance=-0.5)
        assert note.importance == 0.0

    def test_tags_default_empty(self, mem):
        note = mem.add("no tags")
        assert note.tags == []


class TestGet:
    def test_get_existing(self, mem):
        note = mem.add("retrievable")
        fetched = mem.get(note.note_id)
        assert fetched is not None
        assert fetched.note_id == note.note_id

    def test_get_missing_returns_none(self, mem):
        assert mem.get("nonexistent-id") is None


class TestEviction:
    def test_evict_lowest_removes_min_importance(self, mem):
        mem.add("low", importance=0.1)
        mem.add("high", importance=0.9)
        evicted = mem.evict_lowest()
        assert evicted is not None
        assert evicted.importance == pytest.approx(0.1)
        assert mem.size() == 1

    def test_evict_on_full_capacity(self):
        small_mem = ZettelkastenMemory(max_notes=2, similarity_threshold=1.0)
        small_mem.add("a", importance=0.3)
        small_mem.add("b", importance=0.9)
        small_mem.add("c", importance=0.5)
        assert small_mem.size() == 2

    def test_evict_empty_returns_none(self, mem):
        assert mem.evict_lowest() is None


class TestLink:
    def test_manual_link(self, mem):
        n1 = mem.add("note one", tags=["x"])
        n2 = mem.add("note two", tags=["y"])
        mem.link(n1.note_id, n2.note_id)
        assert n2.note_id in mem.get(n1.note_id).links
        assert n1.note_id in mem.get(n2.note_id).links

    def test_link_idempotent(self, mem):
        n1 = mem.add("a", tags=["t"])
        n2 = mem.add("b", tags=["t"])
        mem.link(n1.note_id, n2.note_id)
        mem.link(n1.note_id, n2.note_id)
        assert mem.get(n1.note_id).links.count(n2.note_id) == 1

    def test_link_missing_id_noop(self, mem):
        n1 = mem.add("real note")
        mem.link(n1.note_id, "ghost-id")
        assert mem.get(n1.note_id).links == []

    def test_auto_link_on_tag_overlap(self):
        m = ZettelkastenMemory(max_notes=100, similarity_threshold=0.5)
        n1 = m.add("first", tags=["ml", "nlp"])
        n2 = m.add("second", tags=["ml", "nlp"])
        assert n1.note_id in n2.links
        assert n2.note_id in n1.links


class TestGetLinked:
    def test_returns_linked_notes(self, mem):
        n1 = mem.add("alpha", tags=["x"])
        n2 = mem.add("beta", tags=["y"])
        mem.link(n1.note_id, n2.note_id)
        linked = mem.get_linked(n1.note_id)
        assert any(n.note_id == n2.note_id for n in linked)

    def test_missing_note_returns_empty(self, mem):
        assert mem.get_linked("bad-id") == []


class TestSearch:
    def test_returns_list(self, mem):
        mem.add("alpha content", tags=["alpha"])
        results = mem.search("alpha")
        assert isinstance(results, list)

    def test_top_k_respected(self, mem):
        for i in range(8):
            mem.add(f"note {i}", tags=["common"])
        results = mem.search("common", top_k=3)
        assert len(results) <= 3

    def test_substring_match_boosts_rank(self, mem):
        mem.add("irrelevant note about cats", tags=["animals"])
        target = mem.add("transformers architecture", tags=["ml"], importance=0.5)
        results = mem.search("transformers architecture")
        assert results[0].note_id == target.note_id

    def test_empty_memory_search(self, mem):
        assert mem.search("anything") == []


class TestListAll:
    def test_list_all_empty(self, mem):
        assert mem.list_all() == []

    def test_list_all_count(self, mem):
        mem.add("one")
        mem.add("two")
        mem.add("three")
        assert len(mem.list_all()) == 3
