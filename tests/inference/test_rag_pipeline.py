"""Tests for the RAG pipeline (rag_pipeline.py) — spec-required 12 tests."""

from __future__ import annotations

import torch
import pytest

from src.inference.rag_pipeline import (
    RAGConfig,
    chunk_text,
    BM25Index,
    reciprocal_rank_fusion,
    DenseRetriever,
    RAGPipeline,
)


# ---------------------------------------------------------------------------
# RAGConfig tests
# ---------------------------------------------------------------------------

def test_ragconfig_defaults():
    """RAGConfig should have the specified default values."""
    cfg = RAGConfig()
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32
    assert cfg.n_retrieve == 5
    assert cfg.n_rerank == 3
    assert cfg.fusion_alpha == 0.5
    assert cfg.max_context_len == 1024


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    """chunk_text should split text into multiple word-level chunks."""
    text = " ".join([f"word{i}" for i in range(100)])  # 100 words
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk.split()) <= 20


def test_chunk_text_overlap_creates_shared_words():
    """Consecutive chunks should share words (overlap > 0)."""
    text = " ".join([f"w{i}" for i in range(50)])
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    # The last 3 words of chunk[0] should appear at the start of chunk[1]
    words0 = chunks[0].split()
    words1 = chunks[1].split()
    shared = words0[-3:]
    assert words1[:3] == shared, (
        f"Expected overlap words {shared} at start of chunk[1], got {words1[:3]}"
    )


def test_chunk_text_empty_returns_empty_list():
    """chunk_text on empty string should return []."""
    assert chunk_text("", chunk_size=10, overlap=2) == []


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

def test_bm25_search_returns_list_of_tuples():
    """BM25Index.search should return a list of (int, float) tuples."""
    index = BM25Index()
    docs = ["the quick brown fox", "jumped over the lazy dog", "hello world"]
    index.index(docs)
    results = index.search("fox", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        idx, score = item
        assert isinstance(idx, int)
        assert isinstance(score, float)


def test_bm25_search_exact_query_term_high_score():
    """Document containing the exact query term should rank first."""
    index = BM25Index()
    docs = [
        "machine learning is a subfield of artificial intelligence",
        "the cat sat on the mat in the garden",
        "neural networks are used in deep learning machine tasks",
    ]
    index.index(docs)
    results = index.search("cat sat mat", top_k=3)
    # The second document (index 1) contains all query terms
    top_idx = results[0][0]
    assert top_idx == 1, f"Expected doc 1 at top, got {top_idx}"


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_merges_correctly():
    """RRF should promote docs that appear high in multiple lists."""
    list1 = [0, 1, 2, 3]
    list2 = [1, 0, 3, 2]
    fused = reciprocal_rank_fusion([list1, list2])
    # Doc 0 is rank-1 in list1 and rank-2 in list2
    # Doc 1 is rank-2 in list1 and rank-1 in list2
    # Both should be near the top; all input docs should be present
    assert set(fused) == {0, 1, 2, 3}
    # The top two should be 0 and 1 (highest combined RRF scores)
    assert set(fused[:2]) == {0, 1}


def test_reciprocal_rank_fusion_single_list():
    """RRF on a single ranked list should preserve the original order."""
    ranking = [3, 1, 4, 0, 5]
    fused = reciprocal_rank_fusion([ranking])
    # All docs should appear; order should match the single list (rank 1 → highest score)
    assert set(fused) == set(ranking)
    assert fused[0] == 3  # rank-1 item gets the highest RRF score


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------

def test_dense_retriever_encode_output_shape():
    """DenseRetriever.encode should return a (N, embed_dim) tensor."""
    retriever = DenseRetriever(embed_dim=64)
    texts = ["hello world", "foo bar baz", "test document"]
    embeddings = retriever.encode(texts)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, 64)


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

def test_rag_pipeline_index_documents_runs_without_error():
    """index_documents should complete without raising an exception."""
    cfg = RAGConfig(chunk_size=10, chunk_overlap=2, n_retrieve=2, n_rerank=2)
    pipeline = RAGPipeline(cfg)
    docs = [
        "The quick brown fox jumped over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data to generalise.",
        "Python is a popular programming language used in data science and AI research.",
    ]
    # Should not raise
    pipeline.index_documents(docs)


def test_rag_pipeline_retrieve_returns_list_of_strings():
    """retrieve should return a list of strings (chunk texts)."""
    cfg = RAGConfig(chunk_size=10, chunk_overlap=2, n_retrieve=3, n_rerank=2)
    pipeline = RAGPipeline(cfg)
    docs = [
        "The quick brown fox jumped over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data.",
        "Python is a popular programming language used in data science.",
    ]
    pipeline.index_documents(docs)
    results = pipeline.retrieve("machine learning python")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_rag_pipeline_format_context_contains_query():
    """format_context should include the query in the returned string."""
    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    query = "what is machine learning?"
    chunks = ["Machine learning is a subset of AI.", "It uses data to train models."]
    context = pipeline.format_context(query, chunks)
    assert query in context
    assert "Context:" in context
    assert "Query:" in context
