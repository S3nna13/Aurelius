"""Tests for src/reasoning/scratchpad.py"""

from __future__ import annotations

from src.reasoning.scratchpad import SCRATCHPAD, ScratchEntry, Scratchpad

# ---------- ScratchEntry ----------


class TestScratchEntry:
    def test_auto_id_generated(self):
        e = ScratchEntry(tag="t", content="c")
        assert e.id is not None

    def test_id_is_8_chars(self):
        e = ScratchEntry(tag="t", content="c")
        assert len(e.id) == 8

    def test_id_is_hex(self):
        e = ScratchEntry(tag="t", content="c")
        int(e.id, 16)

    def test_unique_ids(self):
        e1 = ScratchEntry(tag="t", content="c")
        e2 = ScratchEntry(tag="t", content="c")
        assert e1.id != e2.id

    def test_default_pinned_false(self):
        e = ScratchEntry(tag="t", content="c")
        assert e.pinned is False

    def test_pinned_true(self):
        e = ScratchEntry(tag="t", content="c", pinned=True)
        assert e.pinned is True

    def test_tag_stored(self):
        e = ScratchEntry(tag="mytag", content="c")
        assert e.tag == "mytag"

    def test_content_stored(self):
        e = ScratchEntry(tag="t", content="hello world")
        assert e.content == "hello world"


# ---------- Scratchpad.write ----------


class TestScratchpadWrite:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_returns_scratch_entry(self):
        result = self.sp.write("tag", "content")
        assert isinstance(result, ScratchEntry)

    def test_adds_to_read_all(self):
        self.sp.write("tag", "content")
        assert len(self.sp.read_all()) == 1

    def test_pin_flag(self):
        entry = self.sp.write("tag", "content", pin=True)
        assert entry.pinned is True

    def test_multiple_writes(self):
        self.sp.write("a", "1")
        self.sp.write("b", "2")
        self.sp.write("c", "3")
        assert len(self.sp.read_all()) == 3

    def test_tag_stored(self):
        entry = self.sp.write("mytag", "content")
        assert entry.tag == "mytag"

    def test_content_stored(self):
        entry = self.sp.write("tag", "my content")
        assert entry.content == "my content"


# ---------- Scratchpad.read ----------


class TestScratchpadRead:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_returns_matching_entries(self):
        self.sp.write("foo", "a")
        self.sp.write("foo", "b")
        self.sp.write("bar", "c")
        results = self.sp.read("foo")
        assert len(results) == 2

    def test_case_insensitive(self):
        self.sp.write("FOO", "a")
        results = self.sp.read("foo")
        assert len(results) == 1

    def test_no_match_returns_empty(self):
        self.sp.write("bar", "content")
        assert self.sp.read("nonexistent") == []

    def test_returns_correct_content(self):
        self.sp.write("tag", "hello")
        results = self.sp.read("tag")
        assert results[0].content == "hello"


# ---------- Scratchpad.erase ----------


class TestScratchpadErase:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_erase_returns_true_for_valid_id(self):
        entry = self.sp.write("tag", "content")
        assert self.sp.erase(entry.id) is True

    def test_erase_removes_entry(self):
        entry = self.sp.write("tag", "content")
        self.sp.erase(entry.id)
        assert len(self.sp.read_all()) == 0

    def test_erase_returns_false_for_unknown_id(self):
        assert self.sp.erase("deadbeef") is False

    def test_erase_pinned_without_force(self):
        entry = self.sp.write("tag", "content", pin=True)
        result = self.sp.erase(entry.id)
        assert result is False

    def test_erase_pinned_with_force(self):
        entry = self.sp.write("tag", "content", pin=True)
        result = self.sp.erase(entry.id, force=True)
        assert result is True

    def test_erase_pinned_with_force_removes(self):
        entry = self.sp.write("tag", "content", pin=True)
        self.sp.erase(entry.id, force=True)
        assert len(self.sp.read_all()) == 0


# ---------- Scratchpad.clear_unpinned ----------


class TestClearUnpinned:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_removes_non_pinned(self):
        self.sp.write("a", "1")
        self.sp.write("b", "2")
        self.sp.clear_unpinned()
        assert len(self.sp.read_all()) == 0

    def test_returns_count(self):
        self.sp.write("a", "1")
        self.sp.write("b", "2")
        count = self.sp.clear_unpinned()
        assert count == 2

    def test_preserves_pinned(self):
        self.sp.write("a", "1", pin=True)
        self.sp.write("b", "2")
        self.sp.clear_unpinned()
        assert len(self.sp.read_all()) == 1

    def test_preserved_is_pinned(self):
        self.sp.write("a", "1", pin=True)
        self.sp.write("b", "2")
        self.sp.clear_unpinned()
        assert self.sp.read_all()[0].pinned is True

    def test_empty_scratchpad_count(self):
        count = self.sp.clear_unpinned()
        assert count == 0


# ---------- Scratchpad.tags ----------


class TestTags:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_unique_tags(self):
        self.sp.write("foo", "1")
        self.sp.write("foo", "2")
        self.sp.write("bar", "3")
        assert sorted(self.sp.tags()) == ["bar", "foo"]

    def test_ordered_by_first_appearance(self):
        self.sp.write("alpha", "1")
        self.sp.write("beta", "2")
        self.sp.write("alpha", "3")
        tags = self.sp.tags()
        assert tags[0] == "alpha"
        assert tags[1] == "beta"

    def test_empty_scratchpad(self):
        assert self.sp.tags() == []


# ---------- Scratchpad.render ----------


class TestRender:
    def setup_method(self):
        self.sp = Scratchpad()

    def test_non_empty_when_entries_present(self):
        self.sp.write("tag", "content")
        assert len(self.sp.render()) > 0

    def test_pinned_entry_contains_star(self):
        self.sp.write("tag", "content", pin=True)
        assert "★" in self.sp.render()

    def test_unpinned_no_star(self):
        self.sp.write("tag", "content", pin=False)
        assert "★" not in self.sp.render()

    def test_contains_tag(self):
        self.sp.write("mytag", "mycontent")
        assert "mytag" in self.sp.render()

    def test_contains_content(self):
        self.sp.write("tag", "my awesome content")
        assert "my awesome content" in self.sp.render()

    def test_empty_scratchpad(self):
        assert self.sp.render() == ""


# ---------- max_entries eviction ----------


class TestMaxEntries:
    def test_eviction_on_write(self):
        sp = Scratchpad(max_entries=3)
        for i in range(5):
            sp.write("tag", str(i))
        assert len(sp.read_all()) == 3

    def test_eviction_removes_oldest_unpinned(self):
        sp = Scratchpad(max_entries=3)
        e1 = sp.write("tag", "first")
        sp.write("tag", "second")
        sp.write("tag", "third")
        # 4th write should evict first
        sp.write("tag", "fourth")
        ids = [e.id for e in sp.read_all()]
        assert e1.id not in ids

    def test_pinned_not_evicted(self):
        sp = Scratchpad(max_entries=3)
        pinned = sp.write("tag", "pinned", pin=True)
        sp.write("tag", "a")
        sp.write("tag", "b")
        sp.write("tag", "c")  # triggers eviction of unpinned
        ids = [e.id for e in sp.read_all()]
        assert pinned.id in ids


# ---------- SCRATCHPAD singleton ----------


class TestScratchpadSingleton:
    def test_exists(self):
        assert SCRATCHPAD is not None

    def test_is_scratchpad(self):
        assert isinstance(SCRATCHPAD, Scratchpad)
