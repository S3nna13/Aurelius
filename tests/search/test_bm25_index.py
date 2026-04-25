"""Tests for BM25Index — at least 20 tests covering all specified requirements."""
from __future__ import annotations

import math

import pytest

from src.search.bm25_index import BM25Index, _MAX_DOC_ID_LEN, _MAX_TEXT_LEN, _MAX_TOP_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh() -> BM25Index:
    """Return a brand-new BM25Index instance."""
    return BM25Index()


# ---------------------------------------------------------------------------
# 1. Basic add + query: relevant doc comes first
# ---------------------------------------------------------------------------

def test_basic_query_returns_relevant_doc_first():
    idx = fresh()
    idx.add("py", "python programming language")
    idx.add("cook", "delicious recipes and cooking techniques")
    results = idx.query("python programming")
    assert results[0][0] == "py"


# ---------------------------------------------------------------------------
# 2. Score formula verification on minimal 2-doc corpus
# ---------------------------------------------------------------------------

def test_score_formula_minimal_corpus():
    """
    Corpus: doc_a="hello world", doc_b="world"
    Query token: "world"
    Hand-calc with k1=1.5, b=0.75, avgdl=1.5
    N=2, df("world")=2
    IDF = ln((2-2+0.5)/(2+0.5)+1) = ln(0.2+1) = ln(1.2)
    doc_a: dl=2, tf=1, denom = 1 + 1.5*(1-0.75+0.75*2/1.5) = 1+1.5*(0.25+1)=1+1.875=2.875
           score_a = ln(1.2) * (1*2.5)/2.875
    doc_b: dl=1, tf=1, denom = 1 + 1.5*(1-0.75+0.75*1/1.5)=1+1.5*(0.25+0.5)=1+1.125=2.125
           score_b = ln(1.2) * (1*2.5)/2.125
    doc_b has higher score because it is shorter (stronger length norm).
    """
    idx = fresh()
    idx.add("doc_a", "hello world")
    idx.add("doc_b", "world")
    results = idx.query("world")
    assert results[0][0] == "doc_b"

    idf = math.log((2 - 2 + 0.5) / (2 + 0.5) + 1)
    avgdl = 1.5
    score_a = idf * (1 * 2.5) / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / avgdl))
    score_b = idf * (1 * 2.5) / (1 + 1.5 * (1 - 0.75 + 0.75 * 1 / avgdl))
    assert abs(results[0][1] - score_b) < 1e-9
    assert abs(results[1][1] - score_a) < 1e-9


# ---------------------------------------------------------------------------
# 3. Remove doc decrements df correctly
# ---------------------------------------------------------------------------

def test_remove_decrements_df():
    idx = fresh()
    idx.add("a", "apple banana")
    idx.add("b", "banana cherry")
    idx.remove("a")
    assert idx._df.get("apple", 0) == 0
    assert idx._df.get("banana", 0) == 1  # still in doc b


# ---------------------------------------------------------------------------
# 4. Duplicate doc_id raises ValueError
# ---------------------------------------------------------------------------

def test_duplicate_doc_id_raises():
    idx = fresh()
    idx.add("dup", "some text")
    with pytest.raises(ValueError, match="already exists"):
        idx.add("dup", "other text")


# ---------------------------------------------------------------------------
# 5. Empty query tokens returns []
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty():
    idx = fresh()
    idx.add("d", "hello world")
    assert idx.query("   ") == []
    assert idx.query("!!!") == []


# ---------------------------------------------------------------------------
# 6. Empty index returns []
# ---------------------------------------------------------------------------

def test_empty_index_returns_empty():
    idx = fresh()
    assert idx.query("anything") == []


# ---------------------------------------------------------------------------
# 7. top_k=0 raises ValueError
# ---------------------------------------------------------------------------

def test_top_k_zero_raises():
    idx = fresh()
    idx.add("d", "hello")
    with pytest.raises(ValueError):
        idx.query("hello", top_k=0)


# ---------------------------------------------------------------------------
# 8. top_k > 1000 raises ValueError
# ---------------------------------------------------------------------------

def test_top_k_exceeds_max_raises():
    idx = fresh()
    idx.add("d", "hello")
    with pytest.raises(ValueError):
        idx.query("hello", top_k=_MAX_TOP_K + 1)


# ---------------------------------------------------------------------------
# 9. doc_id > 512 chars raises ValueError
# ---------------------------------------------------------------------------

def test_doc_id_too_long_raises():
    idx = fresh()
    long_id = "x" * (_MAX_DOC_ID_LEN + 1)
    with pytest.raises(ValueError, match="doc_id exceeds"):
        idx.add(long_id, "some text")


# ---------------------------------------------------------------------------
# 10. text > 1_000_000 chars raises ValueError
# ---------------------------------------------------------------------------

def test_text_too_long_raises():
    idx = fresh()
    long_text = "a " * (_MAX_TEXT_LEN // 2 + 1)  # well over limit in chars
    with pytest.raises(ValueError, match="text exceeds"):
        idx.add("d", long_text)


# ---------------------------------------------------------------------------
# 11. len() and __contains__
# ---------------------------------------------------------------------------

def test_len_and_contains():
    idx = fresh()
    assert len(idx) == 0
    assert "doc1" not in idx
    idx.add("doc1", "hello world")
    assert len(idx) == 1
    assert "doc1" in idx
    idx.add("doc2", "foo bar")
    assert len(idx) == 2
    idx.remove("doc1")
    assert len(idx) == 1
    assert "doc1" not in idx


# ---------------------------------------------------------------------------
# 12. Remove nonexistent raises KeyError
# ---------------------------------------------------------------------------

def test_remove_nonexistent_raises():
    idx = fresh()
    with pytest.raises(KeyError):
        idx.remove("ghost")


# ---------------------------------------------------------------------------
# 13. All-whitespace text raises ValueError (empty tokens after tokenization)
# ---------------------------------------------------------------------------

def test_all_whitespace_text_raises():
    idx = fresh()
    with pytest.raises(ValueError, match="empty after tokenization"):
        idx.add("ws", "     ")


# ---------------------------------------------------------------------------
# 14. Higher b → stronger length normalization
# ---------------------------------------------------------------------------

def test_higher_b_stronger_length_normalization():
    """
    Two docs: short (1 token) and long (10 tokens of the same word).
    Query = that word.  With high b the short doc should outscore long doc
    more strongly than with low b.
    """
    word = "test"
    short_text = word
    long_text = " ".join([word] * 10)

    idx_low_b = BM25Index(k1=1.5, b=0.0)
    idx_low_b.add("short", short_text)
    idx_low_b.add("long", long_text)

    idx_high_b = BM25Index(k1=1.5, b=1.0)
    idx_high_b.add("short", short_text)
    idx_high_b.add("long", long_text)

    def score_diff(idx: BM25Index) -> float:
        results = {doc_id: sc for doc_id, sc in idx.query(word)}
        return results["short"] - results["long"]

    # With b=1.0 the short doc advantage is more pronounced
    assert score_diff(idx_high_b) > score_diff(idx_low_b)


# ---------------------------------------------------------------------------
# 15. Multiple tokens in query accumulate scores
# ---------------------------------------------------------------------------

def test_multiple_query_tokens_accumulate():
    idx = fresh()
    idx.add("both", "python machine learning")
    idx.add("one", "python only stuff here")
    # "both" should score higher for a 2-token query
    results = idx.query("machine learning")
    assert results[0][0] == "both"


# ---------------------------------------------------------------------------
# 16. Deterministic ordering with equal scores
# ---------------------------------------------------------------------------

def test_deterministic_ordering():
    """Two docs with identical text → sorted output must be stable / consistent."""
    idx = fresh()
    idx.add("aaa", "hello world")
    idx.add("bbb", "hello world")
    r1 = idx.query("hello world")
    r2 = idx.query("hello world")
    assert r1 == r2


# ---------------------------------------------------------------------------
# 17. Query token absent from all docs yields zero scores
# ---------------------------------------------------------------------------

def test_query_token_not_in_corpus():
    idx = fresh()
    idx.add("d", "apple banana cherry")
    results = idx.query("zzzmissing")
    # Should return results but all scores should be 0.0
    assert all(sc == 0.0 for _, sc in results)


# ---------------------------------------------------------------------------
# 18. top_k limits returned results
# ---------------------------------------------------------------------------

def test_top_k_limits_results():
    idx = fresh()
    for i in range(10):
        idx.add(f"doc{i}", f"document number {i} with some unique word token{i}")
    results = idx.query("document number", top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 19. Metadata is stored and accessible
# ---------------------------------------------------------------------------

def test_metadata_stored():
    idx = fresh()
    meta = {"source": "wiki", "year": 2024}
    idx.add("m", "some important text", metadata=meta)
    assert idx._metadata["m"] == meta


# ---------------------------------------------------------------------------
# 20. k1 parameter affects saturation
# ---------------------------------------------------------------------------

def test_k1_affects_score_magnitude():
    """Higher k1 means tf has more influence; a multi-occurrence term matters more."""
    text_repeat = "python python python python python"
    text_single = "python java c rust go"

    idx_low = BM25Index(k1=0.5, b=0.75)
    idx_low.add("repeat", text_repeat)
    idx_low.add("single", text_single)

    idx_high = BM25Index(k1=3.0, b=0.75)
    idx_high.add("repeat", text_repeat)
    idx_high.add("single", text_single)

    def gap(idx: BM25Index) -> float:
        r = dict(idx.query("python"))
        return r["repeat"] - r["single"]

    # Higher k1 → repeated term is rewarded more → larger gap
    assert gap(idx_high) > gap(idx_low)


# ---------------------------------------------------------------------------
# 21. add accepts exactly _MAX_DOC_ID_LEN chars (boundary)
# ---------------------------------------------------------------------------

def test_doc_id_exactly_max_len_accepted():
    idx = fresh()
    exact_id = "x" * _MAX_DOC_ID_LEN
    idx.add(exact_id, "valid text")
    assert exact_id in idx


# ---------------------------------------------------------------------------
# 22. After removing all docs index is empty
# ---------------------------------------------------------------------------

def test_index_empty_after_remove_all():
    idx = fresh()
    idx.add("a", "hello")
    idx.add("b", "world")
    idx.remove("a")
    idx.remove("b")
    assert len(idx) == 0
    assert idx.query("hello") == []
