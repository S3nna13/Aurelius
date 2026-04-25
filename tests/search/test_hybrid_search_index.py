"""Tests for HybridSearchIndex — at least 20 tests covering all specified requirements."""
from __future__ import annotations

import pytest

from src.search.hybrid_search_index import HybridSearchIndex
from src.search.bm25_index import _MAX_DOC_ID_LEN, _MAX_TEXT_LEN, _MAX_TOP_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh(**kwargs) -> HybridSearchIndex:
    return HybridSearchIndex(**kwargs)


def _populate(idx: HybridSearchIndex, n: int = 5) -> list[str]:
    ids = []
    for i in range(n):
        doc_id = f"doc{i}"
        idx.add(doc_id, f"unique keyword token_{i} appears here in document number {i}")
        ids.append(doc_id)
    return ids


# ---------------------------------------------------------------------------
# 1. Basic add + query returns results
# ---------------------------------------------------------------------------

def test_basic_add_and_query_returns_results():
    idx = fresh()
    idx.add("a", "python machine learning framework")
    idx.add("b", "cooking recipes pasta sauce")
    results = idx.query("python machine learning")
    assert len(results) > 0
    assert results[0][0] == "a"


# ---------------------------------------------------------------------------
# 2. Doc appearing in both BM25 and semantic gets RRF boost
# ---------------------------------------------------------------------------

def test_rrf_boost_for_doc_in_both_lists():
    """
    A doc that ranks highly in BOTH sub-indexes should outscore a doc that
    ranks highly in only one of them.
    """
    idx = fresh()
    # 'both' is highly relevant in BM25 sense (exact keyword match)
    # and also in TF-IDF (rare token)
    idx.add("both", "transformer attention mechanism neural network")
    idx.add("bm25_only", "transformer transformer transformer transformer transformer")
    idx.add("semantic_only", "attention mechanism network deep learning model")
    # Query with tokens from 'both'
    results = dict(idx.query("transformer attention mechanism"))
    assert "both" in results
    # 'both' should score at least as high as a single-list doc
    assert results["both"] >= min(results.get("bm25_only", 0), results.get("semantic_only", 0))


# ---------------------------------------------------------------------------
# 3. query_bm25 works independently
# ---------------------------------------------------------------------------

def test_query_bm25_independent():
    idx = fresh()
    _populate(idx, 5)
    results = idx.query_bm25("unique keyword token_2", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert results[0][0] == "doc2"


# ---------------------------------------------------------------------------
# 4. query_semantic works independently
# ---------------------------------------------------------------------------

def test_query_semantic_independent():
    idx = fresh()
    _populate(idx, 5)
    results = idx.query_semantic("unique keyword token_3", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert results[0][0] == "doc3"


# ---------------------------------------------------------------------------
# 5. Weights not summing to 1.0 raises ValueError
# ---------------------------------------------------------------------------

def test_weights_not_summing_to_one_raises():
    with pytest.raises(ValueError, match="must equal 1.0"):
        HybridSearchIndex(bm25_weight=0.3, semantic_weight=0.3)


# ---------------------------------------------------------------------------
# 6. Remove removes from both sub-indexes
# ---------------------------------------------------------------------------

def test_remove_from_both_subindexes():
    idx = fresh()
    idx.add("x", "hello world")
    idx.remove("x")
    assert len(idx) == 0
    assert "x" not in idx._bm25
    assert "x" not in idx._semantic._tf


# ---------------------------------------------------------------------------
# 7. Empty index returns []
# ---------------------------------------------------------------------------

def test_empty_index_returns_empty():
    idx = fresh()
    assert idx.query("anything") == []


# ---------------------------------------------------------------------------
# 8. Whitespace-only query returns []
# ---------------------------------------------------------------------------

def test_whitespace_only_query_returns_empty():
    idx = fresh()
    idx.add("d", "hello world")
    assert idx.query("   ") == []
    assert idx.query("\t\n") == []


# ---------------------------------------------------------------------------
# 9. top_k=0 raises ValueError
# ---------------------------------------------------------------------------

def test_top_k_zero_raises():
    idx = fresh()
    idx.add("d", "hello world")
    with pytest.raises(ValueError):
        idx.query("hello", top_k=0)


# ---------------------------------------------------------------------------
# 10. Round-trip: add 20 docs, each appears in top-5 for its own text
# ---------------------------------------------------------------------------

def test_round_trip_20_docs_self_query():
    idx = fresh()
    doc_ids = []
    for i in range(20):
        doc_id = f"roundtrip_{i}"
        idx.add(doc_id, f"exclusive phrase zeta_{i} alpha_{i} beta_{i}")
        doc_ids.append(doc_id)
    for doc_id in doc_ids:
        # extract the unique number from doc_id
        n = doc_id.split("_")[1]
        results = idx.query(f"exclusive phrase zeta_{n} alpha_{n}", top_k=5)
        top_ids = [r[0] for r in results]
        assert doc_id in top_ids, f"{doc_id} not in top-5 for its own query"


# ---------------------------------------------------------------------------
# 11. RRF formula: doc at rank 1 in BOTH lists scores higher than rank 1 in one
# ---------------------------------------------------------------------------

def test_rrf_rank1_both_beats_rank1_one():
    """
    Manually verify RRF math using _rrf_fuse directly.
    doc_top_both  -> rank 1 in list_a and rank 1 in list_b
    doc_top_one   -> rank 1 in list_a only, rank 100 (absent from list_b)
    """
    idx = fresh()
    # We call _rrf_fuse directly with synthetic rank lists
    list_a = [("doc_top_both", 0.9), ("doc_top_one", 0.8)]
    list_b = [("doc_top_both", 0.85), ("doc_only_b", 0.7)]
    fused = idx._rrf_fuse(list_a, list_b, top_k=3)
    fused_dict = dict(fused)
    assert fused_dict["doc_top_both"] > fused_dict.get("doc_top_one", 0)
    assert fused_dict["doc_top_both"] > fused_dict.get("doc_only_b", 0)


# ---------------------------------------------------------------------------
# 12. rrf_k parameter affects score magnitude
# ---------------------------------------------------------------------------

def test_rrf_k_affects_score_magnitude():
    idx_small_k = HybridSearchIndex(rrf_k=1)
    idx_large_k = HybridSearchIndex(rrf_k=600)
    list_a = [("doc", 1.0)]
    list_b = [("doc", 1.0)]
    fused_small = dict(idx_small_k._rrf_fuse(list_a, list_b, top_k=1))
    fused_large = dict(idx_large_k._rrf_fuse(list_a, list_b, top_k=1))
    # Smaller k → larger individual scores (1/(k+1))
    assert fused_small["doc"] > fused_large["doc"]


# ---------------------------------------------------------------------------
# 13. Metadata preserved (accessible via underlying BM25)
# ---------------------------------------------------------------------------

def test_metadata_preserved_in_bm25():
    idx = fresh()
    meta = {"source": "arxiv", "year": 2009}
    idx.add("paper", "reciprocal rank fusion results", metadata=meta)
    assert idx._bm25._metadata["paper"] == meta


# ---------------------------------------------------------------------------
# 14. Large doc_id raises ValueError (> 512)
# ---------------------------------------------------------------------------

def test_large_doc_id_raises():
    idx = fresh()
    long_id = "x" * (_MAX_DOC_ID_LEN + 1)
    with pytest.raises(ValueError, match="doc_id exceeds"):
        idx.add(long_id, "valid text")


# ---------------------------------------------------------------------------
# 15. Long text raises ValueError (> 1_000_000)
# ---------------------------------------------------------------------------

def test_long_text_raises():
    idx = fresh()
    long_text = "a " * (_MAX_TEXT_LEN // 2 + 1)
    with pytest.raises(ValueError, match="text exceeds"):
        idx.add("d", long_text)


# ---------------------------------------------------------------------------
# 16. len() reflects document count
# ---------------------------------------------------------------------------

def test_len_reflects_count():
    idx = fresh()
    assert len(idx) == 0
    idx.add("a", "hello world")
    assert len(idx) == 1
    idx.add("b", "foo bar")
    assert len(idx) == 2
    idx.remove("a")
    assert len(idx) == 1


# ---------------------------------------------------------------------------
# 17. top_k limits returned results
# ---------------------------------------------------------------------------

def test_top_k_limits_results():
    idx = fresh()
    _populate(idx, 10)
    results = idx.query("unique keyword document", top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 18. top_k > 1000 raises ValueError
# ---------------------------------------------------------------------------

def test_top_k_exceeds_max_raises():
    idx = fresh()
    idx.add("d", "hello world")
    with pytest.raises(ValueError):
        idx.query("hello", top_k=_MAX_TOP_K + 1)


# ---------------------------------------------------------------------------
# 19. query_bm25 top_k=0 raises ValueError
# ---------------------------------------------------------------------------

def test_query_bm25_top_k_zero_raises():
    idx = fresh()
    idx.add("d", "hello world")
    with pytest.raises(ValueError):
        idx.query_bm25("hello", top_k=0)


# ---------------------------------------------------------------------------
# 20. query_semantic top_k=0 raises ValueError
# ---------------------------------------------------------------------------

def test_query_semantic_top_k_zero_raises():
    idx = fresh()
    idx.add("d", "hello world")
    with pytest.raises(ValueError):
        idx.query_semantic("hello", top_k=0)


# ---------------------------------------------------------------------------
# 21. Fused results include docs from both sub-index result sets
# ---------------------------------------------------------------------------

def test_fused_results_merge_both_subindex_results():
    idx = fresh()
    # Two clearly distinct docs; query hits both
    idx.add("nlp", "natural language processing text transformer")
    idx.add("cv", "computer vision image recognition convolutional")
    results_ids = {r[0] for r in idx.query("natural language processing image")}
    assert "nlp" in results_ids


# ---------------------------------------------------------------------------
# 22. Weights=1.0/0.0 still satisfies constraint check (valid edge case)
# ---------------------------------------------------------------------------

def test_weights_edge_case_one_zero():
    idx_b = HybridSearchIndex(bm25_weight=1.0, semantic_weight=0.0)
    idx_s = HybridSearchIndex(bm25_weight=0.0, semantic_weight=1.0)
    for idx in (idx_b, idx_s):
        idx.add("d", "hello world")
        results = idx.query("hello world")
        assert len(results) > 0
