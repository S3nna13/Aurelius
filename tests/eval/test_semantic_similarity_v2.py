"""Tests for src/eval/semantic_similarity_v2.py

Pure PyTorch only.  Uses tiny tensors (small D, B) for speed.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.eval.semantic_similarity_v2 import (
    SemanticSimilarityScorer,
    SimilarityConfig,
    compute_cosine_similarity,
    compute_pairwise_similarity,
    compute_retrieval_metrics,
    embedding_alignment_score,
    embedding_uniformity_score,
    pool_hidden_states,
)

# ---------------------------------------------------------------------------
# Tiny dims used throughout
# ---------------------------------------------------------------------------
B, T, D = 3, 5, 8
N = 4


# ---------------------------------------------------------------------------
# 1. SimilarityConfig defaults
# ---------------------------------------------------------------------------


def test_similarity_config_defaults():
    cfg = SimilarityConfig()
    assert cfg.pooling == "mean"
    assert cfg.normalize is True
    assert cfg.similarity_metric == "cosine"


# ---------------------------------------------------------------------------
# 2. pool_hidden_states — mean pooling shape (B, D)
# ---------------------------------------------------------------------------


def test_pool_hidden_states_mean_shape():
    hidden = torch.randn(B, T, D)
    out = pool_hidden_states(hidden, mode="mean")
    assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# 3. pool_hidden_states — cls pooling returns first token
# ---------------------------------------------------------------------------


def test_pool_hidden_states_cls_returns_first_token():
    hidden = torch.randn(B, T, D)
    out = pool_hidden_states(hidden, mode="cls")
    assert out.shape == (B, D)
    assert torch.allclose(out, hidden[:, 0, :])


# ---------------------------------------------------------------------------
# 4. pool_hidden_states — max pooling shape (B, D)
# ---------------------------------------------------------------------------


def test_pool_hidden_states_max_shape():
    hidden = torch.randn(B, T, D)
    out = pool_hidden_states(hidden, mode="max")
    assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# 5. pool_hidden_states — mean with mask ignores masked positions
# ---------------------------------------------------------------------------


def test_pool_hidden_states_mean_with_mask_ignores_padded():
    torch.manual_seed(0)
    # B=2 sequences, T=4 tokens, D=6
    hidden = torch.randn(2, 4, 6)
    # Only first 2 tokens are real for both sequences.
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.float)

    out = pool_hidden_states(hidden, attention_mask=mask, mode="mean")
    # Expected: mean of first 2 tokens only.
    expected = hidden[:, :2, :].mean(dim=1)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. compute_cosine_similarity — shape (B,)
# ---------------------------------------------------------------------------


def test_compute_cosine_similarity_shape():
    a = torch.randn(B, D)
    b = torch.randn(B, D)
    out = compute_cosine_similarity(a, b)
    assert out.shape == (B,)


# ---------------------------------------------------------------------------
# 7. compute_cosine_similarity — identical vectors = 1.0
# ---------------------------------------------------------------------------


def test_compute_cosine_similarity_identical_vectors():
    a = torch.randn(B, D)
    sims = compute_cosine_similarity(a, a)
    assert torch.allclose(sims, torch.ones(B), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. compute_pairwise_similarity — shape (N, N)
# ---------------------------------------------------------------------------


def test_compute_pairwise_similarity_shape():
    embs = torch.randn(N, D)
    mat = compute_pairwise_similarity(embs, metric="cosine")
    assert mat.shape == (N, N)


# ---------------------------------------------------------------------------
# 9. compute_pairwise_similarity — diagonal = 1.0 for cosine
# ---------------------------------------------------------------------------


def test_compute_pairwise_similarity_diagonal_cosine():
    embs = torch.randn(N, D)
    mat = compute_pairwise_similarity(embs, metric="cosine")
    diag = mat.diag()
    assert torch.allclose(diag, torch.ones(N), atol=1e-5)


# ---------------------------------------------------------------------------
# 10. embedding_alignment_score — in [-1, 1]
# ---------------------------------------------------------------------------


def test_embedding_alignment_score_range():
    src = torch.randn(N, D)
    tgt = torch.randn(N, D)
    score = embedding_alignment_score(src, tgt)
    assert isinstance(score, float)
    assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 11. embedding_uniformity_score — returns finite float
# ---------------------------------------------------------------------------


def test_embedding_uniformity_score_finite():
    embs = F.normalize(torch.randn(N, D), p=2, dim=-1)
    score = embedding_uniformity_score(embs)
    assert isinstance(score, float)
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 12. SemanticSimilarityScorer.encode — shape (B, D)
# ---------------------------------------------------------------------------


def test_scorer_encode_shape():
    cfg = SimilarityConfig(pooling="mean", normalize=False)
    scorer = SemanticSimilarityScorer(cfg)
    hidden = torch.randn(B, T, D)
    out = scorer.encode(hidden)
    assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# 13. SemanticSimilarityScorer.encode — normalized when normalize=True
# ---------------------------------------------------------------------------


def test_scorer_encode_normalized():
    cfg = SimilarityConfig(pooling="mean", normalize=True)
    scorer = SemanticSimilarityScorer(cfg)
    hidden = torch.randn(B, T, D)
    out = scorer.encode(hidden)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# 14. SemanticSimilarityScorer.similarity — shape (B,)
# ---------------------------------------------------------------------------


def test_scorer_similarity_shape():
    cfg = SimilarityConfig()
    scorer = SemanticSimilarityScorer(cfg)
    ha = torch.randn(B, T, D)
    hb = torch.randn(B, T, D)
    out = scorer.similarity(ha, hb)
    assert out.shape == (B,)


# ---------------------------------------------------------------------------
# 15. compute_retrieval_metrics — recall@1 = 1.0 when embeddings are identical
# ---------------------------------------------------------------------------


def test_retrieval_metrics_identical_embeddings_recall_1():
    embs = F.normalize(torch.randn(N, D), p=2, dim=-1)
    # Query and key embeddings are the same — each query matches itself perfectly.
    result = compute_retrieval_metrics(embs, embs, top_k=1)
    assert "recall@k" in result
    assert "mrr" in result
    assert abs(result["recall@k"] - 1.0) < 1e-6
    assert abs(result["mrr"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 16. compute_retrieval_metrics — recall@k <= 1 and mrr in (0, 1]
# ---------------------------------------------------------------------------


def test_retrieval_metrics_random_embeddings_bounds():
    torch.manual_seed(7)
    queries = F.normalize(torch.randn(N, D), p=2, dim=-1)
    keys = F.normalize(torch.randn(N, D), p=2, dim=-1)
    result = compute_retrieval_metrics(queries, keys, top_k=2)
    assert 0.0 <= result["recall@k"] <= 1.0
    assert 0.0 < result["mrr"] <= 1.0


# ---------------------------------------------------------------------------
# 17. SimilarityConfig rejects invalid pooling
# ---------------------------------------------------------------------------


def test_similarity_config_invalid_pooling():
    with pytest.raises(ValueError):
        SimilarityConfig(pooling="last")


# ---------------------------------------------------------------------------
# 18. SimilarityConfig rejects invalid metric
# ---------------------------------------------------------------------------


def test_similarity_config_invalid_metric():
    with pytest.raises(ValueError):
        SimilarityConfig(similarity_metric="euclidean")


# ---------------------------------------------------------------------------
# 19. compute_pairwise_similarity — dot product matrix shape and symmetry
# ---------------------------------------------------------------------------


def test_compute_pairwise_similarity_dot_symmetric():
    embs = torch.randn(N, D)
    mat = compute_pairwise_similarity(embs, metric="dot")
    assert mat.shape == (N, N)
    assert torch.allclose(mat, mat.T, atol=1e-5)


# ---------------------------------------------------------------------------
# 20. embedding_uniformity_score — more uniform = more negative
# ---------------------------------------------------------------------------


def test_embedding_uniformity_score_uniform_vs_clustered():
    # Uniform embeddings on hypersphere should be more negative than clustered ones.
    torch.manual_seed(42)
    uniform_embs = F.normalize(torch.randn(16, 8), p=2, dim=-1)

    # Clustered: all embeddings nearly identical.
    clustered_base = F.normalize(torch.randn(1, 8), p=2, dim=-1)
    noise = torch.randn(16, 8) * 0.001
    clustered_embs = F.normalize(clustered_base + noise, p=2, dim=-1)

    u_uniform = embedding_uniformity_score(uniform_embs)
    u_clustered = embedding_uniformity_score(clustered_embs)

    # Uniform distribution should have lower (more negative) uniformity score.
    assert u_uniform < u_clustered


# ---------------------------------------------------------------------------
# 21. pool_hidden_states — max pooling respects mask
# ---------------------------------------------------------------------------


def test_pool_hidden_states_max_respects_mask():
    # Create hidden states where token 2 and 3 have very large values.
    hidden = torch.zeros(1, 4, 4)
    hidden[0, 2, :] = 100.0  # This should be ignored (masked out)
    hidden[0, 3, :] = 100.0  # This should be ignored (masked out)
    hidden[0, 0, :] = 1.0
    hidden[0, 1, :] = 2.0

    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
    out = pool_hidden_states(hidden, attention_mask=mask, mode="max")
    # Max of real tokens should be 2.0, not 100.0.
    assert torch.allclose(out, torch.full((1, 4), 2.0))
