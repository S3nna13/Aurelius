"""Tests for src/inference/rag_reranker_v2.py.

Covers CrossEncoderScorer, BiEncoderRetriever, RecipRankFusion, and RAGReranker.
Uses d_model=16, N_docs=20 throughout.
"""

from __future__ import annotations

import torch
import pytest

from aurelius.inference.rag_reranker_v2 import (
    CrossEncoderScorer,
    BiEncoderRetriever,
    Document,
    RAGReranker,
    RecipRankFusion,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
N_DOCS = 20


@pytest.fixture()
def scorer() -> CrossEncoderScorer:
    return CrossEncoderScorer(d_model=D_MODEL, hidden_size=64)


@pytest.fixture()
def corpus():
    """Returns (doc_embeddings, doc_ids)."""
    torch.manual_seed(42)
    embs = torch.randn(N_DOCS, D_MODEL)
    ids = list(range(N_DOCS))
    return embs, ids


@pytest.fixture()
def retriever(corpus):
    embs, ids = corpus
    return BiEncoderRetriever(doc_embeddings=embs, doc_ids=ids)


@pytest.fixture()
def query_emb():
    torch.manual_seed(7)
    return torch.randn(D_MODEL)


@pytest.fixture()
def rag_reranker(retriever, scorer):
    return RAGReranker(
        retriever=retriever,
        scorer=scorer,
        top_k_retrieve=10,
        top_k_rerank=5,
    )


# ---------------------------------------------------------------------------
# CrossEncoderScorer tests
# ---------------------------------------------------------------------------

def test_cross_encoder_output_shape(scorer, query_emb, corpus):
    """score() should return a 1-D tensor of length N."""
    embs, _ = corpus
    out = scorer.score(query_emb, embs)
    assert out.shape == (N_DOCS,), f"Expected ({N_DOCS},), got {out.shape}"


def test_cross_encoder_output_finite(scorer, query_emb, corpus):
    """All scores should be finite (no NaN / Inf)."""
    embs, _ = corpus
    out = scorer.score(query_emb, embs)
    assert torch.all(torch.isfinite(out)), "Scores contain non-finite values"


def test_cross_encoder_different_queries_produce_different_scores(scorer, corpus):
    """Two distinct query embeddings should produce different score vectors."""
    embs, _ = corpus
    q1 = torch.randn(D_MODEL)
    q2 = torch.randn(D_MODEL)
    scores1 = scorer.score(q1, embs)
    scores2 = scorer.score(q2, embs)
    assert not torch.allclose(scores1, scores2), (
        "Different queries should produce different scores"
    )


# ---------------------------------------------------------------------------
# BiEncoderRetriever tests
# ---------------------------------------------------------------------------

def test_retriever_returns_documents(retriever, query_emb):
    """retrieve() should return a non-empty list of Document objects."""
    docs = retriever.retrieve(query_emb, top_k=5)
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_retriever_count_lte_top_k(retriever, query_emb):
    """Number of retrieved docs should not exceed top_k."""
    top_k = 7
    docs = retriever.retrieve(query_emb, top_k=top_k)
    assert len(docs) <= top_k


def test_retriever_sorted_by_score_descending(retriever, query_emb):
    """Retrieved docs should be ordered by score descending."""
    docs = retriever.retrieve(query_emb, top_k=10)
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True), (
        "Retrieved docs are not sorted by score descending"
    )


def test_document_score_is_float(retriever, query_emb):
    """Document.score should be a Python float."""
    docs = retriever.retrieve(query_emb, top_k=3)
    for doc in docs:
        assert isinstance(doc.score, float), f"score is {type(doc.score)}, expected float"


# ---------------------------------------------------------------------------
# RecipRankFusion tests
# ---------------------------------------------------------------------------

def test_rrf_fuse_returns_list_of_ints():
    """fuse() should return a list of integers."""
    rrf = RecipRankFusion(k=60)
    result = rrf.fuse([[0, 1, 2], [2, 0, 3]])
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_rrf_fuse_with_scores_correct_length():
    """fuse_with_scores() should cover the union of all doc ids."""
    rrf = RecipRankFusion(k=60)
    lists = [[0, 1, 2], [2, 3, 4], [1, 5]]
    result = rrf.fuse_with_scores(lists)
    expected_ids = {0, 1, 2, 3, 4, 5}
    returned_ids = {doc_id for doc_id, _ in result}
    assert returned_ids == expected_ids
    assert len(result) == len(expected_ids)


def test_rrf_score_decreases_with_rank():
    """A document appearing at rank 1 should beat rank 2 (single list)."""
    rrf = RecipRankFusion(k=60)
    ranked_list = [10, 20, 30]  # 10 is rank-1, 20 rank-2, 30 rank-3
    result = rrf.fuse_with_scores([ranked_list])
    # result is already sorted descending; verify id order matches input rank order
    returned_ids = [doc_id for doc_id, _ in result]
    assert returned_ids == ranked_list, (
        "fuse_with_scores should preserve rank-order for a single list"
    )


def test_rrf_single_list():
    """fuse() should handle a single ranked list without error."""
    rrf = RecipRankFusion(k=60)
    ranked = [5, 3, 1, 2, 4]
    result = rrf.fuse([ranked])
    assert result == ranked, "Single-list RRF should return the same order"


# ---------------------------------------------------------------------------
# RAGReranker tests
# ---------------------------------------------------------------------------

def test_rag_reranker_returns_documents(rag_reranker, query_emb):
    """retrieve_and_rerank() should return a list of Document objects."""
    docs = rag_reranker.retrieve_and_rerank(query_emb)
    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)


def test_rag_reranker_length_lte_top_k_rerank(rag_reranker, query_emb):
    """Output length should not exceed top_k_rerank."""
    docs = rag_reranker.retrieve_and_rerank(query_emb)
    assert len(docs) <= rag_reranker.top_k_rerank


def test_score_documents_output_shape(rag_reranker, retriever, query_emb):
    """score_documents() should return a 1-D tensor matching number of docs."""
    candidates = retriever.retrieve(query_emb, top_k=8)
    scores = rag_reranker.score_documents(query_emb, candidates)
    assert scores.shape == (len(candidates),), (
        f"Expected ({len(candidates)},), got {scores.shape}"
    )


def test_score_documents_finite(rag_reranker, retriever, query_emb):
    """score_documents() should return only finite values."""
    candidates = retriever.retrieve(query_emb, top_k=8)
    scores = rag_reranker.score_documents(query_emb, candidates)
    assert torch.all(torch.isfinite(scores)), "score_documents returned non-finite values"
