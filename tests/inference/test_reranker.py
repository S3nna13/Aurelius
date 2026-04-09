"""Tests for the CrossEncoderReranker module (src/inference/reranker.py)."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.reranker import (
    RerankerConfig,
    ScoredDocument,
    CrossEncoderReranker,
    batch_score,
    format_query_document,
    rerank,
    reciprocal_rank_fusion,
    score_query_document_logit,
    score_query_document_perplexity,
)

# ---------------------------------------------------------------------------
# Shared tiny model config — fast enough to run in CI
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

# Byte-level tokenizer capped at 256 tokens (fits within TINY_CFG.vocab_size)
def byte_encode(s: str) -> list[int]:
    return list(s.encode("utf-8", errors="replace"))[:256]


@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    torch.manual_seed(0)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = RerankerConfig()
    assert cfg.max_seq_len == 256
    assert cfg.score_method == "logit"
    assert cfg.batch_size == 8
    assert cfg.normalize_scores is True


# ---------------------------------------------------------------------------
# 2. test_format_query_document
# ---------------------------------------------------------------------------

def test_format_query_document():
    result = format_query_document("What is AI?", "AI is a field of computer science.")
    assert "Query:" in result
    assert "Document:" in result
    assert "Relevant:" in result


# ---------------------------------------------------------------------------
# 3. test_score_logit_finite
# ---------------------------------------------------------------------------

def test_score_logit_finite(tiny_model):
    score = score_query_document_logit(tiny_model, byte_encode, "hi", "hello world")
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 4. test_score_perplexity_negative
# ---------------------------------------------------------------------------

def test_score_perplexity_negative(tiny_model):
    score = score_query_document_perplexity(tiny_model, byte_encode, "hi", "hello world")
    # Returns -perplexity, so must be <= 0
    assert score <= 0.0
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 5. test_batch_score_length
# ---------------------------------------------------------------------------

def test_batch_score_length(tiny_model):
    cfg = RerankerConfig(score_method="logit")
    documents = ["doc one", "doc two", "doc three"]
    scores = batch_score(tiny_model, byte_encode, "query", documents, cfg)
    assert len(scores) == len(documents)


# ---------------------------------------------------------------------------
# 6. test_batch_score_finite
# ---------------------------------------------------------------------------

def test_batch_score_finite(tiny_model):
    cfg = RerankerConfig(score_method="logit")
    documents = ["alpha", "beta", "gamma"]
    scores = batch_score(tiny_model, byte_encode, "q", documents, cfg)
    for s in scores:
        assert math.isfinite(s)


# ---------------------------------------------------------------------------
# 7. test_rerank_sorted_descending
# ---------------------------------------------------------------------------

def test_rerank_sorted_descending():
    documents = ["doc a", "doc b", "doc c"]
    scores = [0.3, 0.9, 0.1]
    result = rerank("query", documents, scores)
    for i in range(len(result) - 1):
        assert result[i].rerank_score >= result[i + 1].rerank_score


# ---------------------------------------------------------------------------
# 8. test_rerank_original_ranks
# ---------------------------------------------------------------------------

def test_rerank_original_ranks():
    documents = ["first", "second", "third"]
    scores = [0.5, 0.8, 0.2]
    result = rerank("q", documents, scores)
    # original_rank must be the index in the *input* list (0, 1, or 2)
    original_ranks = {sd.original_rank for sd in result}
    assert original_ranks == {0, 1, 2}
    # The highest-scored document ("second", original_rank=1) should be first
    assert result[0].original_rank == 1
    assert result[0].text == "second"


# ---------------------------------------------------------------------------
# 9. test_reciprocal_rank_fusion_order
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_order():
    # Doc 0 is top-ranked in both lists → should win
    rankings = [[0, 1, 2], [0, 2, 1]]
    fused = reciprocal_rank_fusion(rankings)
    assert fused[0] == 0


# ---------------------------------------------------------------------------
# 10. test_reciprocal_rank_fusion_shape
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_shape():
    rankings = [[0, 1, 2], [1, 2, 3]]
    fused = reciprocal_rank_fusion(rankings)
    unique_docs = {d for r in rankings for d in r}
    assert len(fused) == len(unique_docs)


# ---------------------------------------------------------------------------
# 11. test_cross_encoder_rerank_returns_list
# ---------------------------------------------------------------------------

def test_cross_encoder_rerank_returns_list(tiny_model):
    cfg = RerankerConfig(score_method="logit", normalize_scores=True)
    reranker = CrossEncoderReranker(tiny_model, byte_encode, cfg)
    documents = ["cat", "dog", "fish"]
    result = reranker.rerank("pet", documents)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, ScoredDocument)


# ---------------------------------------------------------------------------
# 12. test_cross_encoder_rerank_count
# ---------------------------------------------------------------------------

def test_cross_encoder_rerank_count(tiny_model):
    cfg = RerankerConfig(score_method="logit")
    reranker = CrossEncoderReranker(tiny_model, byte_encode, cfg)
    documents = ["one", "two", "three", "four"]
    result = reranker.rerank("num", documents)
    assert len(result) == len(documents)


# ---------------------------------------------------------------------------
# 13. test_cross_encoder_ndcg_range
# ---------------------------------------------------------------------------

def test_cross_encoder_ndcg_range(tiny_model):
    cfg = RerankerConfig(score_method="logit")
    reranker = CrossEncoderReranker(tiny_model, byte_encode, cfg)
    documents = ["very relevant doc", "somewhat ok", "totally unrelated"]
    labels = [2, 1, 0]
    ndcg = reranker.evaluate_ndcg("query about docs", documents, labels, top_k=3)
    assert 0.0 <= ndcg <= 1.0


# ---------------------------------------------------------------------------
# 14. test_cross_encoder_ndcg_perfect
# ---------------------------------------------------------------------------

def test_cross_encoder_ndcg_perfect(tiny_model):
    """A perfectly ranked list should yield NDCG = 1.0."""
    cfg = RerankerConfig(score_method="logit")
    reranker = CrossEncoderReranker(tiny_model, byte_encode, cfg)

    documents = ["best", "good", "poor"]
    labels = [2, 1, 0]

    # Patch rerank to return the ideal ordering without running the model
    original_rerank = reranker.rerank

    def perfect_rerank(query, docs, original_scores=None):
        return [
            ScoredDocument("best", 0, 1.0, None),
            ScoredDocument("good", 1, 0.5, None),
            ScoredDocument("poor", 2, 0.0, None),
        ]

    reranker.rerank = perfect_rerank  # type: ignore[method-assign]
    try:
        ndcg = reranker.evaluate_ndcg("query", documents, labels, top_k=3)
    finally:
        reranker.rerank = original_rerank

    assert ndcg == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 15. test_cross_encoder_fusion_rerank
# ---------------------------------------------------------------------------

def test_cross_encoder_fusion_rerank(tiny_model):
    cfg = RerankerConfig(score_method="logit")
    reranker = CrossEncoderReranker(tiny_model, byte_encode, cfg)
    list1 = ["apple", "banana", "cherry"]
    list2 = ["cherry", "date", "apple"]
    result = reranker.fusion_rerank("fruit", [list1, list2])
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)
    # All unique docs should be present
    all_unique = set(list1) | set(list2)
    assert set(result) == all_unique
