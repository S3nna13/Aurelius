"""Tests for src.memory.progressive_search."""

from __future__ import annotations

import time

import pytest

from src.memory.progressive_search import (
    DEFAULT_PROGRESSIVE_SEARCHER,
    PROGRESSIVE_SEARCH_REGISTRY,
    IndexEntry,
    ProgressiveSearchError,
    ProgressiveSearcher,
    SearchResult,
)

# ---------------------------------------------------------------------------
# IndexEntry / SearchResult dataclasses
# ---------------------------------------------------------------------------


def test_index_entry_defaults():
    entry = IndexEntry(
        entry_id="e1",
        summary_tags=["python", "ml"],
        timestamp=time.time(),
        layer=1,
    )
    assert entry.access_count == 0


def test_search_result_fields():
    sr = SearchResult(
        entry_id="e1",
        score=1.5,
        timeline_context=["ctx"],
        full_content="full",
    )
    assert sr.entry_id == "e1"
    assert sr.score == 1.5
    assert sr.timeline_context == ["ctx"]
    assert sr.full_content == "full"


# ---------------------------------------------------------------------------
# ProgressiveSearcher.index_entry & basic search
# ---------------------------------------------------------------------------


def test_index_entry_increases_size():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    assert ps.stats()["index_size"] == 1


def test_search_finds_keyword_match():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["machine", "learning"], 1000.0, 1))
    results = ps.search("machine")
    assert len(results) == 1
    assert results[0].entry_id == "e1"


def test_search_case_insensitive():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["Python", "ML"], 1000.0, 1))
    results = ps.search("python")
    assert len(results) == 1
    assert results[0].entry_id == "e1"


def test_search_multi_term_overlap():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["python", "ml"], 1000.0, 1))
    ps.index_entry(IndexEntry("e2", ["python", "web"], 1001.0, 1))
    ps.index_entry(IndexEntry("e3", ["java", "backend"], 1002.0, 1))
    results = ps.search("python ml")
    entry_ids = [r.entry_id for r in results]
    assert "e1" in entry_ids
    # e1 should rank higher than e2 because it matches both terms
    assert entry_ids[0] == "e1"


def test_search_empty_index_returns_empty():
    ps = ProgressiveSearcher()
    assert ps.search("anything") == []


def test_search_no_match_returns_empty():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    assert ps.search("blockchain") == []


# ---------------------------------------------------------------------------
# Keyword ranking & recency bonus
# ---------------------------------------------------------------------------


def test_search_recency_boosts_newer_entries():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("old", ["python"], 1000.0, 1))
    ps.index_entry(IndexEntry("new", ["python"], 2000.0, 1))
    results = ps.search("python")
    assert results[0].entry_id == "new"
    assert results[1].entry_id == "old"
    assert results[0].score > results[1].score


def test_search_access_count_tiebreak():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("a", ["python"], 1000.0, 1, access_count=5))
    ps.index_entry(IndexEntry("b", ["python"], 1000.0, 1, access_count=10))
    results = ps.search("python")
    # Same overlap, same timestamp → higher access_count wins
    assert results[0].entry_id == "b"
    assert results[1].entry_id == "a"


# ---------------------------------------------------------------------------
# Timeline context (Layer 2)
# ---------------------------------------------------------------------------


def test_timeline_context_includes_neighbors():
    ps = ProgressiveSearcher()
    for i in range(5):
        ps.index_entry(IndexEntry(f"e{i}", [f"tag{i}"], float(i), 1))
    results = ps.search("tag2", timeline_radius=1)
    ctx = results[0].timeline_context
    assert len(ctx) == 2
    assert any("e1" in c for c in ctx)
    assert any("e3" in c for c in ctx)


def test_timeline_context_sorted_by_proximity():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("mid", ["target"], 100.0, 1))
    ps.index_entry(IndexEntry("close", ["nearby"], 101.0, 1))
    ps.index_entry(IndexEntry("far", ["distant"], 200.0, 1))
    results = ps.search("target", timeline_radius=2)
    ctx = results[0].timeline_context
    assert ctx[0].startswith("close")
    assert ctx[1].startswith("far")


def test_timeline_context_excludes_self():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("solo", ["only"], 1.0, 1))
    results = ps.search("only")
    assert results[0].timeline_context == []


# ---------------------------------------------------------------------------
# Full content fetch (Layer 3)
# ---------------------------------------------------------------------------


def test_fetch_full_true_returns_content():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    ps.set_full_content("e1", "Detailed AI content here.")
    results = ps.search("ai", fetch_full=True)
    assert results[0].full_content == "Detailed AI content here."


def test_fetch_full_false_returns_none():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    ps.set_full_content("e1", "Detailed AI content here.")
    results = ps.search("ai", fetch_full=False)
    assert results[0].full_content is None


def test_fetch_full_missing_content_returns_none():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    results = ps.search("ai", fetch_full=True)
    assert results[0].full_content is None


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def test_remove_deletes_entry_and_content():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["ai"], 1000.0, 1))
    ps.set_full_content("e1", "content")
    assert ps.remove("e1") is True
    assert ps.search("ai") == []
    assert ps.stats()["full_content_store_size"] == 0


def test_remove_unknown_returns_false():
    ps = ProgressiveSearcher()
    assert ps.remove("ghost") is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_empty():
    ps = ProgressiveSearcher()
    s = ps.stats()
    assert s == {
        "index_size": 0,
        "avg_tags_per_entry": 0.0,
        "full_content_store_size": 0,
        "avg_access_count": 0.0,
    }


def test_stats_with_entries():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["a", "b"], 1000.0, 1, access_count=4))
    ps.index_entry(IndexEntry("e2", ["c"], 1001.0, 1, access_count=2))
    ps.set_full_content("e1", "content")
    s = ps.stats()
    assert s["index_size"] == 2
    assert s["avg_tags_per_entry"] == 1.5
    assert s["full_content_store_size"] == 1
    assert s["avg_access_count"] == 3.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_negative_top_k_raises():
    ps = ProgressiveSearcher()
    with pytest.raises(ProgressiveSearchError):
        ps.search("q", top_k=-1)


def test_negative_timeline_radius_raises():
    ps = ProgressiveSearcher()
    with pytest.raises(ProgressiveSearchError):
        ps.search("q", timeline_radius=-1)


# ---------------------------------------------------------------------------
# Singleton & registry
# ---------------------------------------------------------------------------


def test_default_searcher_singleton():
    assert isinstance(DEFAULT_PROGRESSIVE_SEARCHER, ProgressiveSearcher)


def test_registry_contains_default():
    assert "default" in PROGRESSIVE_SEARCH_REGISTRY
    assert PROGRESSIVE_SEARCH_REGISTRY["default"] is DEFAULT_PROGRESSIVE_SEARCHER


# ---------------------------------------------------------------------------
# Overwrite behavior
# ---------------------------------------------------------------------------


def test_index_entry_overwrites():
    ps = ProgressiveSearcher()
    ps.index_entry(IndexEntry("e1", ["old"], 1000.0, 1))
    ps.index_entry(IndexEntry("e1", ["new"], 1001.0, 2))
    assert ps.stats()["index_size"] == 1
    results = ps.search("new")
    assert len(results) == 1
    assert results[0].entry_id == "e1"
    results_old = ps.search("old")
    assert len(results_old) == 0
