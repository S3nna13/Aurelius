"""Tests: plugins/memory/zettelkasten_memory.py — Auto-linking graph memory with importance-based eviction."""

from __future__ import annotations

import pytest

from plugins.memory.zettelkasten_memory import (
    ZettelkastenMemory,
    ZettelNote,
)


@pytest.fixture
def zettel():
    return ZettelkastenMemory(max_notes=100, similarity_threshold=0.7)


class TestZettelNote:
    def test_note_fields(self):
        note = ZettelNote(
            note_id="n1",
            content="test content",
            tags=["tag1", "tag2"],
            importance=0.8,
            created_at=1234.0,
            links=["n2"],
        )
        assert note.note_id == "n1"
        assert note.content == "test content"
        assert note.tags == ["tag1", "tag2"]
        assert note.importance == 0.8
        assert note.created_at == 1234.0
        assert note.links == ["n2"]

    def test_note_default_links(self):
        note = ZettelNote("n1", "c", [], 0.5, 0.0)
        assert note.links == []


class TestZettelkastenMemory:
    def test_add_generates_id(self, zettel):
        note = zettel.add("content", tags=["tag"])
        assert note.note_id is not None
        assert note.content == "content"

    def test_add_clips_importance(self, zettel):
        note = zettel.add("c", importance=1.5)
        assert note.importance == 1.0
        note2 = zettel.add("c2", importance=-0.5)
        assert note2.importance == 0.0

    def test_add_auto_links_by_tag_overlap(self, zettel):
        zettel.add("note1", tags=["python", "ai"])
        note2 = zettel.add("note2", tags=["python", "ml"])
        assert note2.note_id != list(zettel._notes.values())[0].note_id

    def test_add_eviction_when_full(self):
        zettel = ZettelkastenMemory(max_notes=2, similarity_threshold=0.0)
        n1 = zettel.add("note1", importance=0.5)
        _n2 = zettel.add("note2", importance=0.9)
        assert zettel.size() == 2
        _n3 = zettel.add("note3", importance=0.1)  # lowest importance = n1 evicted
        assert zettel.size() == 2
        assert n1.note_id not in zettel._notes

    def test_get_found(self, zettel):
        note = zettel.add("test content")
        retrieved = zettel.get(note.note_id)
        assert retrieved is note

    def test_get_not_found(self, zettel):
        assert zettel.get("ghost") is None

    def test_search_by_content_substring(self, zettel):
        zettel.add("apple pie recipe")
        zettel.add("orange juice")
        zettel.add("apple watch review")
        results = zettel.search("apple")
        assert len(results) >= 2

    def test_search_by_tag(self, zettel):
        zettel.add("note a", tags=["python", "ai"])
        zettel.add("note b", tags=["rust", "ml"])
        results = zettel.search("python")
        assert len(results) >= 1
        assert results[0].note_id == zettel._notes.keys().__iter__().__next__()

    def test_search_top_k(self, zettel):
        for i in range(10):
            zettel.add(f"content {i}", importance=i / 10.0)
        results = zettel.search("content", top_k=3)
        assert len(results) == 3

    def test_link(self, zettel):
        n1 = zettel.add("note1")
        n2 = zettel.add("note2")
        zettel.link(n1.note_id, n2.note_id)
        linked = zettel.get_linked(n1.note_id)
        assert n2.note_id in [note.note_id for note in linked]

    def test_link_nonexistent_ids(self, zettel):
        zettel.add("existing")
        # Should not raise
        zettel.link("ghost", "also_ghost")

    def test_get_linked_empty(self, zettel):
        note = zettel.add("lonely note")
        assert zettel.get_linked(note.note_id) == []

    def test_list_all(self, zettel):
        zettel.add("a")
        zettel.add("b")
        zettel.add("c")
        all_notes = zettel.list_all()
        assert len(all_notes) == 3

    def test_size(self, zettel):
        assert zettel.size() == 0
        zettel.add("a")
        zettel.add("b")
        assert zettel.size() == 2

    def test_evict_lowest(self, zettel):
        zettel.add("low", importance=0.1)
        zettel.add("high", importance=0.9)
        evicted = zettel.evict_lowest()
        assert evicted is not None
        assert zettel.size() == 1
        assert evicted.importance == 0.1

    def test_evict_lowest_empty(self, zettel):
        assert zettel.evict_lowest() is None
