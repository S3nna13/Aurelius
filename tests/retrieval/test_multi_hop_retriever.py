"""Tests for src/retrieval/multi_hop_retriever.py (10+ tests)."""

from __future__ import annotations

import pytest

from src.retrieval.multi_hop_retriever import (
    HopResult,
    MultiHopConfig,
    MultiHopRetriever,
    RETRIEVAL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Minimal stub retriever
# ---------------------------------------------------------------------------

class _StubRetriever:
    """Returns fixed (doc_id, score) pairs regardless of query."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._results[:k]


# ---------------------------------------------------------------------------
# 1. Single hop when top score below threshold
# ---------------------------------------------------------------------------
def test_single_hop_below_threshold():
    stub = _StubRetriever([("doc_a", 0.3), ("doc_b", 0.2)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=3, top_k_per_hop=2, expansion_threshold=0.5)
    results = mhr.retrieve("what is AI?", cfg)
    assert len(results) == 1
    assert results[0].hop == 1


# ---------------------------------------------------------------------------
# 2. Multiple hops when score meets threshold
# ---------------------------------------------------------------------------
def test_multiple_hops():
    stub = _StubRetriever([("doc_x", 0.9), ("doc_y", 0.8)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=3, top_k_per_hop=2, expansion_threshold=0.5)
    results = mhr.retrieve("deep learning", cfg)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 3. max_hops respected
# ---------------------------------------------------------------------------
def test_max_hops_respected():
    stub = _StubRetriever([("doc_1", 1.0)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=2, top_k_per_hop=1, expansion_threshold=0.0)
    results = mhr.retrieve("query", cfg)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# 4. HopResult fields correct
# ---------------------------------------------------------------------------
def test_hop_result_fields():
    stub = _StubRetriever([("doc_a", 0.7)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=1, top_k_per_hop=1, expansion_threshold=0.5)
    results = mhr.retrieve("hello", cfg)
    hr = results[0]
    assert hr.hop == 1
    assert hr.query == "hello"
    assert hr.retrieved_ids == ["doc_a"]
    assert hr.scores == [0.7]


# ---------------------------------------------------------------------------
# 5. Query expands between hops
# ---------------------------------------------------------------------------
def test_query_expands():
    stub = _StubRetriever([("long_snippet_doc_id_here", 0.9)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=2, top_k_per_hop=1, expansion_threshold=0.5)
    results = mhr.retrieve("initial", cfg)
    assert len(results) == 2
    # Second hop query should contain original query text
    assert "initial" in results[1].query


# ---------------------------------------------------------------------------
# 6. flatten returns unique ids
# ---------------------------------------------------------------------------
def test_flatten_unique():
    # Both hops return same doc
    stub = _StubRetriever([("doc_shared", 0.9)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=2, top_k_per_hop=1, expansion_threshold=0.5)
    results = mhr.retrieve("q", cfg)
    flat = mhr.flatten(results)
    assert len(flat) == len(set(flat))


# ---------------------------------------------------------------------------
# 7. flatten preserves first-hop ordering
# ---------------------------------------------------------------------------
def test_flatten_first_hop_order():
    stub = _StubRetriever([("a", 0.9), ("b", 0.8), ("c", 0.7)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=1, top_k_per_hop=3, expansion_threshold=0.0)
    results = mhr.retrieve("q", cfg)
    flat = mhr.flatten(results)
    assert flat[0] == "a"


# ---------------------------------------------------------------------------
# 8. flatten empty
# ---------------------------------------------------------------------------
def test_flatten_empty():
    mhr = MultiHopRetriever(_StubRetriever([]))
    assert mhr.flatten([]) == []


# ---------------------------------------------------------------------------
# 9. BM25 index merged with dense
# ---------------------------------------------------------------------------
def test_bm25_merged():
    dense = _StubRetriever([("dense_doc", 0.6)])
    bm25 = _StubRetriever([("bm25_doc", 0.55)])
    mhr = MultiHopRetriever(dense, bm25)
    cfg = MultiHopConfig(max_hops=1, top_k_per_hop=5, expansion_threshold=0.0)
    results = mhr.retrieve("q", cfg)
    assert "dense_doc" in results[0].retrieved_ids
    assert "bm25_doc" in results[0].retrieved_ids


# ---------------------------------------------------------------------------
# 10. top_k_per_hop limits results
# ---------------------------------------------------------------------------
def test_top_k_limits():
    stub = _StubRetriever([("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6)])
    mhr = MultiHopRetriever(stub)
    cfg = MultiHopConfig(max_hops=1, top_k_per_hop=2, expansion_threshold=0.0)
    results = mhr.retrieve("q", cfg)
    assert len(results[0].retrieved_ids) <= 2


# ---------------------------------------------------------------------------
# 11. RETRIEVAL_REGISTRY entry
# ---------------------------------------------------------------------------
def test_registry_entry():
    assert "multi_hop" in RETRIEVAL_REGISTRY
    assert RETRIEVAL_REGISTRY["multi_hop"] is MultiHopRetriever
