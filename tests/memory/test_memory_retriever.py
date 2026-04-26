"""Tests for src.memory.memory_retriever — 20+ tests."""

from __future__ import annotations

import pytest

from src.memory.memory_retriever import (
    _MAX_QUERY_LEN,
    _MAX_TOP_K,
    MEMORY_RETRIEVER,
    MemoryRetriever,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# MemoryRetriever construction
# ---------------------------------------------------------------------------


def test_alpha_below_zero_raises():
    with pytest.raises(ValueError, match="alpha"):
        MemoryRetriever(alpha=-0.01)


def test_alpha_above_one_raises():
    with pytest.raises(ValueError, match="alpha"):
        MemoryRetriever(alpha=1.01)


def test_alpha_zero_valid():
    r = MemoryRetriever(alpha=0.0)
    assert r.alpha == 0.0


def test_alpha_one_valid():
    r = MemoryRetriever(alpha=1.0)
    assert r.alpha == 1.0


# ---------------------------------------------------------------------------
# index + query basic correctness
# ---------------------------------------------------------------------------


def test_index_and_query_basic():
    r = MemoryRetriever()
    r.index("memory1", "the quick brown fox", importance=0.8)
    results = r.query("quick brown")
    assert len(results) >= 1
    assert results[0].key == "memory1"


def test_query_returns_retrieval_result_objects():
    r = MemoryRetriever()
    r.index("k", "some text content", importance=0.5)
    results = r.query("text")
    assert isinstance(results[0], RetrievalResult)


def test_combined_score_in_result():
    r = MemoryRetriever(alpha=0.7)
    r.index("k", "hello world", importance=0.5)
    results = r.query("hello")
    assert results[0].combined_score > 0.0


# ---------------------------------------------------------------------------
# BM25 scoring: more hits → higher score
# ---------------------------------------------------------------------------


def test_more_term_hits_higher_bm25():
    r = MemoryRetriever()
    r.index("sparse", "cat sat on the mat", importance=0.5)
    r.index("dense", "cat cat cat cat cat", importance=0.5)
    results = r.query("cat")
    keys_ordered = [res.key for res in results]
    assert keys_ordered[0] == "dense"


# ---------------------------------------------------------------------------
# alpha weighting
# ---------------------------------------------------------------------------


def test_alpha_zero_score_is_pure_importance():
    r = MemoryRetriever(alpha=0.0)
    r.index("low", "matching text", importance=0.2)
    r.index("high", "matching text", importance=0.9)
    results = r.query("matching text")
    # alpha=0 → combined = 0*bm25_norm + 1*importance
    assert results[0].key == "high"


def test_alpha_one_score_is_pure_bm25():
    r = MemoryRetriever(alpha=1.0)
    r.index("irrelevant", "matching text", importance=0.9)
    r.index("relevant", "matching text matching text", importance=0.1)
    results = r.query("matching")
    # alpha=1 → combined = bm25_norm; higher tf wins
    assert results[0].key == "relevant"


def test_combined_score_formula_exact():
    r = MemoryRetriever(alpha=0.6)
    r.index("only", "unique term xyzzy", importance=0.4)
    results = r.query("xyzzy")
    res = results[0]
    # bm25_norm = 1.0 (only entry); combined = 0.6*1.0 + 0.4*0.4
    expected = 0.6 * 1.0 + 0.4 * 0.4
    assert abs(res.combined_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_entry_not_in_results():
    r = MemoryRetriever()
    r.index("to_remove", "searchable content here", importance=0.5)
    r.remove("to_remove")
    results = r.query("searchable")
    assert all(res.key != "to_remove" for res in results)


def test_remove_nonexistent_key_no_error():
    r = MemoryRetriever()
    r.remove("ghost")  # should not raise


def test_remove_updates_len():
    r = MemoryRetriever()
    r.index("a", "text", importance=0.5)
    r.index("b", "text", importance=0.5)
    r.remove("a")
    assert len(r) == 1


# ---------------------------------------------------------------------------
# Empty index / empty query
# ---------------------------------------------------------------------------


def test_empty_index_returns_empty_list():
    r = MemoryRetriever()
    assert r.query("anything") == []


def test_empty_query_returns_empty_list():
    r = MemoryRetriever()
    r.index("k", "value", importance=0.5)
    # query of only punctuation tokenizes to nothing
    results = r.query("!!! ???")
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_query_too_long_raises():
    r = MemoryRetriever()
    with pytest.raises(ValueError, match="query exceeds"):
        r.query("x" * (_MAX_QUERY_LEN + 1))


def test_top_k_exceeds_max_raises():
    r = MemoryRetriever()
    r.index("k", "v", importance=0.5)
    with pytest.raises(ValueError, match="top_k exceeds"):
        r.query("v", top_k=_MAX_TOP_K + 1)


# ---------------------------------------------------------------------------
# importance_floor filtering
# ---------------------------------------------------------------------------


def test_importance_floor_filters_low_importance():
    r = MemoryRetriever()
    r.index("low_imp", "shared text", importance=0.1)
    r.index("high_imp", "shared text", importance=0.8)
    results = r.query("shared text", importance_floor=0.5)
    keys = {res.key for res in results}
    assert "high_imp" in keys
    assert "low_imp" not in keys


# ---------------------------------------------------------------------------
# Re-index (update) replaces old entry
# ---------------------------------------------------------------------------


def test_reindex_replaces_old_entry():
    r = MemoryRetriever()
    r.index("k", "old content alpha", importance=0.3)
    r.index("k", "new content beta", importance=0.9)
    # After re-index, bm25_score for "alpha" query must be 0 (token not in doc)
    alpha_results = r.query("alpha")
    beta_results = r.query("beta")
    # "alpha" no longer in index tf → bm25_score must be 0 for key "k"
    for res in alpha_results:
        if res.key == "k":
            assert res.bm25_score == 0.0, "old token 'alpha' should not score"
    assert any(res.key == "k" for res in beta_results)
    assert any(res.bm25_score > 0 for res in beta_results if res.key == "k")


def test_reindex_does_not_grow_len():
    r = MemoryRetriever()
    r.index("k", "value1", importance=0.5)
    r.index("k", "value2", importance=0.5)
    assert len(r) == 1


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_zero_initially():
    r = MemoryRetriever()
    assert len(r) == 0


def test_len_after_adds():
    r = MemoryRetriever()
    r.index("a", "text a", importance=0.5)
    r.index("b", "text b", importance=0.5)
    assert len(r) == 2


# ---------------------------------------------------------------------------
# Adversarial: special chars in key
# ---------------------------------------------------------------------------


def test_special_chars_in_key_indexed_correctly():
    r = MemoryRetriever()
    key = "key/with:special#chars"
    r.index(key, "searchable value", importance=0.5)
    results = r.query("searchable")
    assert any(res.key == key for res in results)


# ---------------------------------------------------------------------------
# Round-trip: 20 entries, each self-query in top-1
# ---------------------------------------------------------------------------


def test_round_trip_self_query_top_1():
    """Each entry, when queried by its unique term, should appear in top-1."""
    r = MemoryRetriever()
    # Use unique token per entry to guarantee discriminability
    entries = {f"entry_{i}": f"uniquetoken_{i} shared filler text" for i in range(20)}
    for key, val in entries.items():
        r.index(key, val, importance=0.5)
    for i in range(20):
        key = f"entry_{i}"
        results = r.query(f"uniquetoken_{i}")
        assert len(results) >= 1, f"no results for entry_{i}"
        assert results[0].key == key, f"expected {key} at top-1, got {results[0].key}"


# ---------------------------------------------------------------------------
# MEMORY_RETRIEVER singleton
# ---------------------------------------------------------------------------


def test_memory_retriever_singleton_exists():
    assert isinstance(MEMORY_RETRIEVER, MemoryRetriever)
