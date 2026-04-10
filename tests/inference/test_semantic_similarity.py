"""Tests for src/inference/semantic_similarity.py."""

import pytest
import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.semantic_similarity import (
    SimilarityConfig,
    pool_hidden_states,
    get_embeddings,
    cosine_similarity,
    dot_similarity,
    euclidean_distance,
    compute_similarity,
    SemanticSimilarityModel,
)

# ---------------------------------------------------------------------------
# Shared fixtures
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


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. SimilarityConfig defaults
# ---------------------------------------------------------------------------

def test_similarity_config_defaults():
    cfg = SimilarityConfig()
    assert cfg.pooling == "mean"
    assert cfg.normalize is True
    assert cfg.similarity_metric == "cosine"


def test_similarity_config_custom():
    cfg = SimilarityConfig(pooling="last", normalize=False, similarity_metric="dot")
    assert cfg.pooling == "last"
    assert cfg.normalize is False
    assert cfg.similarity_metric == "dot"


# ---------------------------------------------------------------------------
# 2. pool_hidden_states - mean pooling shape
# ---------------------------------------------------------------------------

def test_pool_mean_shape():
    B, T, d = 3, 10, 64
    hidden = torch.randn(B, T, d)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = pool_hidden_states(hidden, mask, "mean")
    assert out.shape == (B, d)


def test_pool_mean_no_mask():
    B, T, d = 2, 8, 32
    hidden = torch.randn(B, T, d)
    out = pool_hidden_states(hidden, None, "mean")
    assert out.shape == (B, d)


def test_pool_mean_respects_mask():
    """Mean pooling over a partial mask should differ from full mask."""
    B, T, d = 1, 8, 16
    hidden = torch.randn(B, T, d)
    full_mask = torch.ones(B, T, dtype=torch.bool)
    half_mask = torch.zeros(B, T, dtype=torch.bool)
    half_mask[0, :4] = True
    out_full = pool_hidden_states(hidden, full_mask, "mean")
    out_half = pool_hidden_states(hidden, half_mask, "mean")
    assert not torch.allclose(out_full, out_half)


# ---------------------------------------------------------------------------
# 3. pool_hidden_states - last token pooling shape
# ---------------------------------------------------------------------------

def test_pool_last_shape():
    B, T, d = 4, 12, 48
    hidden = torch.randn(B, T, d)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = pool_hidden_states(hidden, mask, "last")
    assert out.shape == (B, d)


def test_pool_last_selects_correct_token():
    """Last pooling should return the token at the last valid position."""
    B, T, d = 2, 6, 8
    hidden = torch.randn(B, T, d)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = pool_hidden_states(hidden, mask, "last")
    expected = hidden[:, T - 1, :]
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# 4. pool_hidden_states - cls pooling shape
# ---------------------------------------------------------------------------

def test_pool_cls_shape():
    B, T, d = 2, 10, 64
    hidden = torch.randn(B, T, d)
    out = pool_hidden_states(hidden, None, "cls")
    assert out.shape == (B, d)


def test_pool_cls_selects_first_token():
    B, T, d = 3, 5, 16
    hidden = torch.randn(B, T, d)
    out = pool_hidden_states(hidden, None, "cls")
    assert torch.allclose(out, hidden[:, 0, :])


# ---------------------------------------------------------------------------
# 5. get_embeddings - shape and normalization
# ---------------------------------------------------------------------------

def test_get_embeddings_shape(small_model):
    input_ids = torch.randint(0, 256, (2, 8))
    cfg = SimilarityConfig(normalize=False)
    emb = get_embeddings(small_model, input_ids, cfg)
    assert emb.shape == (2, 64)  # (B, d_model)


def test_get_embeddings_normalized(small_model):
    input_ids = torch.randint(0, 256, (3, 8))
    cfg = SimilarityConfig(normalize=True)
    emb = get_embeddings(small_model, input_ids, cfg)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_get_embeddings_not_normalized_when_disabled(small_model):
    input_ids = torch.randint(0, 256, (2, 8))
    cfg = SimilarityConfig(normalize=False)
    emb = get_embeddings(small_model, input_ids, cfg)
    norms = emb.norm(dim=-1)
    # Not expected to all be 1.0 when normalization is off
    assert not torch.allclose(norms, torch.ones(2), atol=1e-3)


def test_get_embeddings_last_pooling(small_model):
    input_ids = torch.randint(0, 256, (2, 8))
    cfg = SimilarityConfig(pooling="last", normalize=False)
    emb = get_embeddings(small_model, input_ids, cfg)
    assert emb.shape == (2, 64)


# ---------------------------------------------------------------------------
# 6. cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    v = torch.randn(4, 32)
    scores = cosine_similarity(v, v)
    assert torch.allclose(scores, torch.ones(4), atol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    a = torch.zeros(1, 4)
    b = torch.zeros(1, 4)
    a[0, 0] = 1.0
    b[0, 1] = 1.0
    score = cosine_similarity(a, b)
    assert torch.allclose(score, torch.zeros(1), atol=1e-5)


def test_cosine_similarity_shape():
    a = torch.randn(5, 16)
    b = torch.randn(5, 16)
    out = cosine_similarity(a, b)
    assert out.shape == (5,)


def test_cosine_similarity_range():
    a = torch.randn(10, 32)
    b = torch.randn(10, 32)
    scores = cosine_similarity(a, b)
    assert (scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# 7. dot_similarity
# ---------------------------------------------------------------------------

def test_dot_similarity_shape():
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    out = dot_similarity(a, b)
    assert out.shape == (4,)


def test_dot_similarity_value():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[2.0, 3.0, 4.0]])
    score = dot_similarity(a, b)
    assert torch.allclose(score, torch.tensor([2.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# 8. euclidean_distance
# ---------------------------------------------------------------------------

def test_euclidean_distance_identical():
    v = torch.randn(3, 16)
    dist = euclidean_distance(v, v)
    assert torch.allclose(dist, torch.zeros(3), atol=1e-5)


def test_euclidean_distance_shape():
    a = torch.randn(6, 32)
    b = torch.randn(6, 32)
    out = euclidean_distance(a, b)
    assert out.shape == (6,)


def test_euclidean_distance_non_negative():
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    dist = euclidean_distance(a, b)
    assert (dist >= 0).all()


# ---------------------------------------------------------------------------
# 9. compute_similarity dispatching
# ---------------------------------------------------------------------------

def test_compute_similarity_dispatches_cosine():
    a = torch.randn(3, 16)
    b = torch.randn(3, 16)
    direct = cosine_similarity(a, b)
    via_dispatch = compute_similarity(a, b, "cosine")
    assert torch.allclose(direct, via_dispatch, atol=1e-6)


def test_compute_similarity_dispatches_dot():
    a = torch.randn(3, 16)
    b = torch.randn(3, 16)
    direct = dot_similarity(a, b)
    via_dispatch = compute_similarity(a, b, "dot")
    assert torch.allclose(direct, via_dispatch, atol=1e-6)


def test_compute_similarity_dispatches_euclidean():
    a = torch.randn(3, 16)
    b = torch.randn(3, 16)
    direct = euclidean_distance(a, b)
    via_dispatch = compute_similarity(a, b, "euclidean")
    assert torch.allclose(direct, via_dispatch, atol=1e-6)


def test_compute_similarity_invalid_metric():
    a = torch.randn(2, 8)
    b = torch.randn(2, 8)
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        compute_similarity(a, b, "unknown_metric")


# ---------------------------------------------------------------------------
# 10. SemanticSimilarityModel
# ---------------------------------------------------------------------------

def test_semantic_similarity_model_embed_shape(small_model):
    sim_model = SemanticSimilarityModel(small_model)
    input_ids = torch.randint(0, 256, (3, 8))
    emb = sim_model.embed(input_ids)
    assert emb.shape == (3, 64)  # (B, d_model)


def test_semantic_similarity_model_embed_unit_norm(small_model):
    sim_model = SemanticSimilarityModel(small_model)
    input_ids = torch.randint(0, 256, (4, 8))
    emb = sim_model.embed(input_ids)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_semantic_similarity_model_similarity_range(small_model):
    """Cosine similarity values must lie within [-1, 1]."""
    cfg = SimilarityConfig(similarity_metric="cosine")
    sim_model = SemanticSimilarityModel(small_model, cfg)
    ids_a = torch.randint(0, 256, (5, 8))
    ids_b = torch.randint(0, 256, (5, 8))
    scores = sim_model.similarity(ids_a, ids_b)
    assert scores.shape == (5,)
    assert (scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all()


def test_semantic_similarity_model_identical_inputs(small_model):
    """Cosine similarity of a sequence with itself should be ~1.0."""
    cfg = SimilarityConfig(similarity_metric="cosine")
    sim_model = SemanticSimilarityModel(small_model, cfg)
    ids = torch.randint(0, 256, (2, 8))
    scores = sim_model.similarity(ids, ids)
    assert torch.allclose(scores, torch.ones(2), atol=1e-4)


def test_semantic_similarity_model_euclidean(small_model):
    """Euclidean distance of a sequence with itself should be ~0."""
    cfg = SimilarityConfig(similarity_metric="euclidean", normalize=False)
    sim_model = SemanticSimilarityModel(small_model, cfg)
    ids = torch.randint(0, 256, (2, 8))
    scores = sim_model.similarity(ids, ids)
    assert torch.allclose(scores, torch.zeros(2), atol=1e-5)


def test_pool_invalid_strategy_raises():
    hidden = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        pool_hidden_states(hidden, None, "max_pool_unsupported")
