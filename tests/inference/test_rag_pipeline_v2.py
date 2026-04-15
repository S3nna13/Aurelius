"""Tests for rag_pipeline_v2.py — 16 tests covering all public API components."""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.rag_pipeline_v2 import (
    RAGConfig,
    Document,
    EmbeddingIndex,
    build_rag_prompt,
    score_document_relevance,
    deduplicate_docs,
    RAGPipeline,
)

DIM = 32

# Tiny mock functions used across tests
_mock_encoder = lambda text: torch.randn(DIM)
_mock_generator = lambda prompt: "Generated response."


# ---------------------------------------------------------------------------
# Helper: build a small index with n_docs random entries
# ---------------------------------------------------------------------------

def _make_index(n: int = 5, dim: int = DIM) -> EmbeddingIndex:
    idx = EmbeddingIndex(dim=dim)
    for i in range(n):
        idx.add(f"doc_{i}", f"Document text number {i}.", torch.randn(dim))
    return idx


# ---------------------------------------------------------------------------
# 1. RAGConfig defaults
# ---------------------------------------------------------------------------

def test_ragconfig_defaults():
    cfg = RAGConfig()
    assert cfg.n_docs == 5
    assert cfg.max_doc_len == 256
    assert cfg.query_prefix == "Query: "
    assert cfg.doc_prefix == "Document: "
    assert cfg.rerank is False
    assert cfg.deduplicate is True


# ---------------------------------------------------------------------------
# 2. Document creation
# ---------------------------------------------------------------------------

def test_document_creation_basic():
    doc = Document(doc_id="d1", text="Hello world")
    assert doc.doc_id == "d1"
    assert doc.text == "Hello world"
    assert doc.embedding is None
    assert doc.score == 0.0


def test_document_creation_with_embedding_and_score():
    emb = torch.randn(DIM)
    doc = Document(doc_id="d2", text="Some text", embedding=emb, score=0.87)
    assert doc.embedding is not None
    assert doc.score == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# 3. EmbeddingIndex add / len
# ---------------------------------------------------------------------------

def test_embedding_index_add_increases_len():
    idx = EmbeddingIndex(dim=DIM)
    assert len(idx) == 0
    idx.add("a", "text a", torch.randn(DIM))
    assert len(idx) == 1
    idx.add("b", "text b", torch.randn(DIM))
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# 4. EmbeddingIndex search returns k docs
# ---------------------------------------------------------------------------

def test_embedding_index_search_returns_k_docs():
    idx = _make_index(n=10)
    query_emb = torch.randn(DIM)
    results = idx.search(query_emb, k=4)
    assert len(results) == 4
    for doc in results:
        assert isinstance(doc, Document)


# ---------------------------------------------------------------------------
# 5. EmbeddingIndex search returns docs sorted by score descending
# ---------------------------------------------------------------------------

def test_embedding_index_search_sorted_by_score_desc():
    idx = _make_index(n=8)
    query_emb = torch.randn(DIM)
    results = idx.search(query_emb, k=5)
    scores = [d.score for d in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results are not sorted descending: {scores}"
    )


# ---------------------------------------------------------------------------
# 6. build_rag_prompt contains query
# ---------------------------------------------------------------------------

def test_build_rag_prompt_contains_query():
    cfg = RAGConfig()
    docs = [Document(doc_id="d1", text="Some retrieved text.")]
    query = "What is the answer?"
    prompt = build_rag_prompt(query, docs, cfg)
    assert query in prompt
    assert cfg.query_prefix in prompt


# ---------------------------------------------------------------------------
# 7. build_rag_prompt contains doc text
# ---------------------------------------------------------------------------

def test_build_rag_prompt_contains_doc_text():
    cfg = RAGConfig()
    doc_text = "This is a document about transformers."
    docs = [Document(doc_id="d1", text=doc_text)]
    prompt = build_rag_prompt("some query", docs, cfg)
    assert doc_text in prompt
    assert cfg.doc_prefix in prompt


# ---------------------------------------------------------------------------
# 8. score_document_relevance is in [-1, 1]
# ---------------------------------------------------------------------------

def test_score_document_relevance_in_range():
    for _ in range(20):
        q = torch.randn(DIM)
        d = torch.randn(DIM)
        sim = score_document_relevance(q, d)
        assert isinstance(sim, float)
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6, f"Cosine similarity out of range: {sim}"


# ---------------------------------------------------------------------------
# 9. deduplicate_docs removes identical embeddings
# ---------------------------------------------------------------------------

def test_deduplicate_docs_removes_identical():
    emb = torch.randn(DIM)
    docs = [
        Document(doc_id="d1", text="text", embedding=emb.clone(), score=0.9),
        Document(doc_id="d2", text="text copy", embedding=emb.clone(), score=0.8),
        Document(doc_id="d3", text="text copy 2", embedding=emb.clone(), score=0.7),
    ]
    result = deduplicate_docs(docs, threshold=0.95)
    # All three are identical; only one should survive
    assert len(result) == 1
    assert result[0].doc_id == "d1"  # highest score kept


# ---------------------------------------------------------------------------
# 10. deduplicate_docs keeps different docs
# ---------------------------------------------------------------------------

def test_deduplicate_docs_keeps_different():
    # Orthogonal embeddings -> cosine sim = 0, well below any reasonable threshold
    e1 = torch.zeros(DIM)
    e1[0] = 1.0
    e2 = torch.zeros(DIM)
    e2[1] = 1.0
    docs = [
        Document(doc_id="d1", text="alpha", embedding=e1, score=0.9),
        Document(doc_id="d2", text="beta", embedding=e2, score=0.8),
    ]
    result = deduplicate_docs(docs, threshold=0.95)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 11. RAGPipeline.encode_query is L2-normalized
# ---------------------------------------------------------------------------

def test_ragpipeline_encode_query_normalized():
    idx = _make_index(n=3)
    pipeline = RAGPipeline(_mock_encoder, _mock_generator, idx, RAGConfig())
    enc = pipeline.encode_query("What is attention?")
    norm = enc.norm().item()
    assert math.isclose(norm, 1.0, abs_tol=1e-5), f"Expected norm 1.0, got {norm}"


# ---------------------------------------------------------------------------
# 12. RAGPipeline.retrieve length <= k
# ---------------------------------------------------------------------------

def test_ragpipeline_retrieve_length_lte_k():
    idx = _make_index(n=5)
    cfg = RAGConfig(n_docs=3, deduplicate=False)
    pipeline = RAGPipeline(_mock_encoder, _mock_generator, idx, cfg)
    docs = pipeline.retrieve("some query", k=3)
    assert len(docs) <= 3


# ---------------------------------------------------------------------------
# 13. RAGPipeline.generate returns tuple (str, list)
# ---------------------------------------------------------------------------

def test_ragpipeline_generate_returns_tuple():
    idx = _make_index(n=5)
    pipeline = RAGPipeline(_mock_encoder, _mock_generator, idx, RAGConfig())
    result = pipeline.generate("What is machine learning?")
    assert isinstance(result, tuple)
    assert len(result) == 2
    response, retrieved_docs = result
    assert isinstance(response, str)
    assert isinstance(retrieved_docs, list)


# ---------------------------------------------------------------------------
# 14. get_retrieval_stats keys present
# ---------------------------------------------------------------------------

def test_get_retrieval_stats_keys_present():
    idx = _make_index(n=4)
    pipeline = RAGPipeline(_mock_encoder, _mock_generator, idx, RAGConfig())
    _, docs = pipeline.generate("test query")
    stats = pipeline.get_retrieval_stats(docs)
    assert "n_docs" in stats
    assert "mean_score" in stats
    assert "max_score" in stats
    assert "min_score" in stats


# ---------------------------------------------------------------------------
# 15. Empty index search returns empty list
# ---------------------------------------------------------------------------

def test_empty_index_search_returns_empty_list():
    idx = EmbeddingIndex(dim=DIM)
    query_emb = torch.randn(DIM)
    results = idx.search(query_emb, k=5)
    assert results == []


# ---------------------------------------------------------------------------
# 16. get_retrieval_stats with empty docs returns zeros
# ---------------------------------------------------------------------------

def test_get_retrieval_stats_empty_docs():
    idx = EmbeddingIndex(dim=DIM)
    pipeline = RAGPipeline(_mock_encoder, _mock_generator, idx, RAGConfig())
    stats = pipeline.get_retrieval_stats([])
    assert stats["n_docs"] == 0.0
    assert stats["mean_score"] == 0.0
    assert stats["max_score"] == 0.0
    assert stats["min_score"] == 0.0
