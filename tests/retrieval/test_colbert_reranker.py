"""Tests for :mod:`src.retrieval.colbert_reranker`."""

from __future__ import annotations

import torch
import pytest

from src.retrieval.colbert_reranker import ColBERTConfig, ColBERTReranker
from src.retrieval.reciprocal_rank_fusion import RankedDoc


def make_doc(doc_id: str, text: str, score: float = 0.5) -> RankedDoc:
    return RankedDoc(doc_id=doc_id, text=text, score=score, source="test")


# ---------------------------------------------------------------------------
# ColBERTConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ColBERTConfig()
    assert cfg.dim == 128
    assert cfg.max_query_len == 32
    assert cfg.max_doc_len == 180


def test_config_custom():
    cfg = ColBERTConfig(dim=64, max_query_len=16, max_doc_len=64)
    assert cfg.dim == 64


# ---------------------------------------------------------------------------
# encode_query / encode_doc shape and normalization
# ---------------------------------------------------------------------------


def test_encode_query_shape():
    rr = ColBERTReranker()
    q = rr.encode_query("what is retrieval augmented generation")
    assert q.dim() == 2
    assert q.shape[1] == 128


def test_encode_query_respects_max_len():
    cfg = ColBERTConfig(max_query_len=3)
    rr = ColBERTReranker(cfg)
    q = rr.encode_query("one two three four five six")
    assert q.shape[0] <= 3


def test_encode_doc_shape():
    rr = ColBERTReranker()
    d = rr.encode_doc("this is a document about machine learning")
    assert d.dim() == 2
    assert d.shape[1] == 128


def test_encode_doc_respects_max_len():
    cfg = ColBERTConfig(max_doc_len=5)
    rr = ColBERTReranker(cfg)
    words = " ".join([f"word{i}" for i in range(20)])
    d = rr.encode_doc(words)
    assert d.shape[0] <= 5


def test_encode_query_l2_normalized():
    rr = ColBERTReranker()
    q = rr.encode_query("test normalization check")
    norms = q.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-5)


def test_encode_doc_l2_normalized():
    rr = ColBERTReranker()
    d = rr.encode_doc("another normalization test document")
    norms = d.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-5)


def test_empty_text_does_not_crash():
    rr = ColBERTReranker()
    q = rr.encode_query("")
    assert q.shape[0] >= 1


# ---------------------------------------------------------------------------
# maxsim
# ---------------------------------------------------------------------------


def test_maxsim_returns_float():
    rr = ColBERTReranker()
    Q = rr.encode_query("query tokens")
    D = rr.encode_doc("document tokens here")
    s = rr.maxsim(Q, D)
    assert isinstance(s, float)


def test_maxsim_identical_query_doc_high_score():
    rr = ColBERTReranker()
    text = "machine learning retrieval"
    Q = rr.encode_query(text)
    D = rr.encode_doc(text)
    s = rr.maxsim(Q, D)
    assert s > 0.0


def test_maxsim_cosine_bounds():
    rr = ColBERTReranker()
    Q = rr.encode_query("hello world")
    D = rr.encode_doc("foo bar baz")
    s = rr.maxsim(Q, D)
    assert -Q.shape[0] - 1 <= s <= Q.shape[0] + 1


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


def test_rerank_returns_sorted_descending():
    rr = ColBERTReranker()
    docs = [
        make_doc("d1", "python programming language"),
        make_doc("d2", "gardening tips for spring"),
        make_doc("d3", "python data science tutorial"),
    ]
    result = rr.rerank("python programming", docs)
    scores = [d.score for d in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_source_field():
    rr = ColBERTReranker()
    docs = [make_doc("d1", "test document")]
    result = rr.rerank("test", docs)
    assert all(d.source == "colbert_reranked" for d in result)


def test_rerank_top_k():
    rr = ColBERTReranker()
    docs = [make_doc(f"d{i}", f"document number {i}") for i in range(10)]
    result = rr.rerank("document", docs, top_k=3)
    assert len(result) == 3


def test_rerank_no_top_k_returns_all():
    rr = ColBERTReranker()
    docs = [make_doc(f"d{i}", f"sample text {i}") for i in range(5)]
    result = rr.rerank("sample", docs)
    assert len(result) == 5


def test_rerank_preserves_doc_id_and_text():
    rr = ColBERTReranker()
    docs = [make_doc("unique_id_42", "very specific unique content abc")]
    result = rr.rerank("unique", docs)
    assert result[0].doc_id == "unique_id_42"
    assert result[0].text == "very specific unique content abc"


def test_rerank_empty_docs_returns_empty():
    rr = ColBERTReranker()
    result = rr.rerank("query", [])
    assert result == []


# ---------------------------------------------------------------------------
# batch_rerank
# ---------------------------------------------------------------------------


def test_batch_rerank_length():
    rr = ColBERTReranker()
    queries = ["python", "java", "rust"]
    docs_per_query = [
        [make_doc(f"d{i}", f"python tutorial {i}") for i in range(3)],
        [make_doc(f"d{i}", f"java guide {i}") for i in range(4)],
        [make_doc(f"d{i}", f"rust programming {i}") for i in range(2)],
    ]
    results = rr.batch_rerank(queries, docs_per_query)
    assert len(results) == 3
    assert len(results[0]) == 3
    assert len(results[1]) == 4
    assert len(results[2]) == 2


def test_batch_rerank_each_result_sorted():
    rr = ColBERTReranker()
    queries = ["search", "index"]
    docs_per_query = [
        [make_doc(f"a{i}", f"search result {i} information") for i in range(5)],
        [make_doc(f"b{i}", f"index entry {i} data") for i in range(5)],
    ]
    results = rr.batch_rerank(queries, docs_per_query)
    for ranked in results:
        scores = [d.score for d in ranked]
        assert scores == sorted(scores, reverse=True)
