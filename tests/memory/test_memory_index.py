"""Tests for src.memory.memory_index."""

from __future__ import annotations

from src.memory.memory_index import MEMORY_INDEX, IndexedEntry, MemoryIndex

# ---------------------------------------------------------------------------
# IndexedEntry
# ---------------------------------------------------------------------------


def test_indexed_entry_fields():
    entry = IndexedEntry(entry_id="abc", tokens=["hello", "world"], content="Hello World")
    assert entry.entry_id == "abc"
    assert entry.tokens == ["hello", "world"]
    assert entry.content == "Hello World"


# ---------------------------------------------------------------------------
# MemoryIndex.index
# ---------------------------------------------------------------------------


def test_index_increases_len():
    mi = MemoryIndex()
    mi.index("e1", "hello world")
    assert len(mi) == 1


def test_index_multiple_entries():
    mi = MemoryIndex()
    mi.index("e1", "hello world")
    mi.index("e2", "goodbye world")
    assert len(mi) == 2


def test_index_stores_entry():
    mi = MemoryIndex()
    mi.index("myid", "sample content here")
    assert len(mi) == 1


def test_index_same_id_overwrites():
    mi = MemoryIndex()
    mi.index("e1", "first content")
    mi.index("e1", "second content")
    # Overwriting should not grow the index
    assert len(mi) == 1


def test_index_tokenizes_lowercase():
    mi = MemoryIndex()
    mi.index("e1", "Hello World")
    results = mi.search("hello")
    assert len(results) == 1


def test_index_tokenizes_on_non_alphanumeric():
    mi = MemoryIndex()
    mi.index("e1", "foo-bar_baz.qux")
    # Tokens should be: foo, bar, baz, qux
    results = mi.search("foo")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# MemoryIndex.search
# ---------------------------------------------------------------------------


def test_search_finds_indexed_content():
    mi = MemoryIndex()
    mi.index("e1", "the quick brown fox")
    results = mi.search("quick")
    assert len(results) == 1


def test_search_returns_entry_id_score_tuples():
    mi = MemoryIndex()
    mi.index("e1", "hello world")
    results = mi.search("hello")
    assert len(results) == 1
    entry_id, score = results[0]
    assert entry_id == "e1"
    assert isinstance(score, float)


def test_search_score_positive_for_match():
    mi = MemoryIndex()
    mi.index("e1", "machine learning model")
    mi.index("e2", "unrelated stuff here")
    results = mi.search("machine learning")
    # Should have at least e1 with positive score
    scores = dict(results)
    assert scores.get("e1", 0.0) > 0.0


def test_search_top_k_limits_results():
    mi = MemoryIndex()
    for i in range(10):
        mi.index(f"e{i}", f"word{i} common term here")
    results = mi.search("common", top_k=3)
    assert len(results) <= 3


def test_search_empty_index_returns_empty():
    mi = MemoryIndex()
    results = mi.search("anything")
    assert results == []


def test_search_no_match_returns_empty():
    mi = MemoryIndex()
    mi.index("e1", "apples and oranges")
    results = mi.search("xyz_no_match")
    assert results == []


def test_search_sorted_descending_by_score():
    mi = MemoryIndex()
    # e1 has "python" twice, e2 has "python" once
    mi.index("e1", "python python programming")
    mi.index("e2", "python programming language")
    mi.index("e3", "java programming language")
    results = mi.search("python")
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_default_is_5():
    mi = MemoryIndex()
    for i in range(10):
        mi.index(f"e{i}", f"common keyword entry {i}")
    results = mi.search("common")
    assert len(results) <= 5


def test_search_multi_token_query():
    mi = MemoryIndex()
    mi.index("e1", "machine learning in python")
    mi.index("e2", "deep learning neural networks")
    mi.index("e3", "python web framework django")
    results = mi.search("python learning")
    # e1 matches both tokens; should appear
    entry_ids = [eid for eid, _ in results]
    assert "e1" in entry_ids


def test_search_case_insensitive():
    mi = MemoryIndex()
    mi.index("e1", "Hello World")
    results = mi.search("HELLO")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# MemoryIndex.remove
# ---------------------------------------------------------------------------


def test_remove_returns_true_for_known_id():
    mi = MemoryIndex()
    mi.index("e1", "some content")
    assert mi.remove("e1") is True


def test_remove_returns_false_for_unknown_id():
    mi = MemoryIndex()
    assert mi.remove("nope") is False


def test_remove_reduces_len():
    mi = MemoryIndex()
    mi.index("e1", "hello")
    mi.index("e2", "world")
    mi.remove("e1")
    assert len(mi) == 1


def test_remove_entry_not_searchable_after():
    mi = MemoryIndex()
    mi.index("e1", "unique secret phrase")
    mi.remove("e1")
    results = mi.search("unique secret")
    assert all(eid != "e1" for eid, _ in results)


def test_remove_only_removes_target():
    mi = MemoryIndex()
    mi.index("e1", "keep this one")
    mi.index("e2", "remove this one")
    mi.remove("e2")
    assert len(mi) == 1
    results = mi.search("keep")
    assert len(results) == 1
    assert results[0][0] == "e1"


def test_remove_updates_inverted_index():
    mi = MemoryIndex()
    mi.index("e1", "shared token doc1")
    mi.index("e2", "shared token doc2")
    mi.remove("e1")
    results = mi.search("shared")
    entry_ids = [eid for eid, _ in results]
    assert "e1" not in entry_ids
    assert "e2" in entry_ids


def test_remove_twice_returns_false_second_time():
    mi = MemoryIndex()
    mi.index("e1", "content")
    mi.remove("e1")
    assert mi.remove("e1") is False


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_zero_initially():
    mi = MemoryIndex()
    assert len(mi) == 0


def test_len_after_index():
    mi = MemoryIndex()
    mi.index("a", "alpha")
    mi.index("b", "beta")
    assert len(mi) == 2


def test_len_after_remove():
    mi = MemoryIndex()
    mi.index("a", "alpha")
    mi.remove("a")
    assert len(mi) == 0


# ---------------------------------------------------------------------------
# MEMORY_INDEX singleton
# ---------------------------------------------------------------------------


def test_memory_index_singleton_exists():
    assert isinstance(MEMORY_INDEX, MemoryIndex)


def test_memory_index_singleton_type():
    assert type(MEMORY_INDEX).__name__ == "MemoryIndex"


def test_memory_index_singleton_usable():
    # Clean-slate mini index for isolation
    mi = MemoryIndex()
    mi.index("test_id", "singleton smoke test")
    assert len(mi) == 1
