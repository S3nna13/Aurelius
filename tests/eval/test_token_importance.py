"""Tests for src/eval/token_importance.py."""

import pytest
import torch

from src.eval.token_importance import (
    TokenImportanceConfig,
    TokenImportanceScorer,
    aggregate_embeddings,
    attention_importance,
    gradient_saliency,
    integrated_gradients,
    normalize_scores,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

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

B = 1
T = 6
D = 64


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, 256, (B, T))


# ---------------------------------------------------------------------------
# Test 1: config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = TokenImportanceConfig()
    assert cfg.method == "gradient"
    assert cfg.n_ig_steps == 20
    assert cfg.ig_baseline == "zero"
    assert cfg.aggregate == "l2"
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# Test 2: aggregate_l2 shape
# ---------------------------------------------------------------------------


def test_aggregate_l2_shape():
    emb = torch.randn(B, T, D)
    out = aggregate_embeddings(emb, "l2")
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: aggregate_l2 non-negative
# ---------------------------------------------------------------------------


def test_aggregate_l2_nonneg():
    emb = torch.randn(B, T, D)
    out = aggregate_embeddings(emb, "l2")
    assert (out >= 0).all(), "L2 norm must be non-negative"


# ---------------------------------------------------------------------------
# Test 4: aggregate_mean shape
# ---------------------------------------------------------------------------


def test_aggregate_mean_shape():
    emb = torch.randn(B, T, D)
    out = aggregate_embeddings(emb, "mean")
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 5: normalize sums to 1
# ---------------------------------------------------------------------------


def test_normalize_scores_sums_to_one():
    scores = torch.rand(B, T)
    normed = normalize_scores(scores)
    row_sums = normed.sum(dim=-1)
    for i, s in enumerate(row_sums):
        assert abs(s.item() - 1.0) < 1e-5, f"Row {i} sum = {s.item()}, expected ~1.0"


# ---------------------------------------------------------------------------
# Test 6: normalize zero row → uniform
# ---------------------------------------------------------------------------


def test_normalize_scores_zero_row():
    scores = torch.zeros(1, T)
    normed = normalize_scores(scores)
    expected = 1.0 / T
    assert normed.shape == (1, T)
    for val in normed[0]:
        assert abs(val.item() - expected) < 1e-6, f"Expected uniform {expected}, got {val.item()}"


# ---------------------------------------------------------------------------
# Test 7: gradient_saliency shape
# ---------------------------------------------------------------------------


def test_gradient_saliency_shape(tiny_model, input_ids):
    grad = gradient_saliency(tiny_model, input_ids, target_position=2, target_token=5)
    assert grad.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {grad.shape}"


# ---------------------------------------------------------------------------
# Test 8: gradient_saliency nonzero
# ---------------------------------------------------------------------------


def test_gradient_saliency_nonzero(tiny_model, input_ids):
    grad = gradient_saliency(tiny_model, input_ids, target_position=2, target_token=5)
    assert not torch.all(grad == 0), "Gradient should not be all zeros"


# ---------------------------------------------------------------------------
# Test 9: integrated_gradients shape
# ---------------------------------------------------------------------------


def test_integrated_gradients_shape(tiny_model, input_ids):
    ig = integrated_gradients(tiny_model, input_ids, target_position=2, target_token=5, n_steps=5)
    assert ig.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {ig.shape}"


# ---------------------------------------------------------------------------
# Test 10: attention_importance shape
# ---------------------------------------------------------------------------


def test_attention_importance_shape(tiny_model, input_ids):
    attn = attention_importance(tiny_model, input_ids, target_position=2)
    assert attn.shape == (B, T), f"Expected ({B}, {T}), got {attn.shape}"


# ---------------------------------------------------------------------------
# Test 11: attention_importance non-negative
# ---------------------------------------------------------------------------


def test_attention_importance_nonneg(tiny_model, input_ids):
    attn = attention_importance(tiny_model, input_ids, target_position=2)
    assert (attn >= 0).all(), "Attention importance values must be non-negative"


# ---------------------------------------------------------------------------
# Test 12: scorer gradient method shape
# ---------------------------------------------------------------------------


def test_scorer_gradient_method_shape(tiny_model, input_ids):
    cfg = TokenImportanceConfig(method="gradient", normalize=False)
    scorer = TokenImportanceScorer(tiny_model, cfg)
    scores = scorer.score(input_ids, target_position=2, target_token=5)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 13: scorer top_k indices
# ---------------------------------------------------------------------------


def test_scorer_top_k_indices(tiny_model, input_ids):
    k = 3
    cfg = TokenImportanceConfig(method="gradient", normalize=False)
    scorer = TokenImportanceScorer(tiny_model, cfg)
    indices = scorer.top_k_tokens(input_ids, target_position=2, target_token=5, k=k)
    assert indices.shape == (B, k), f"Expected ({B}, {k}), got {indices.shape}"
    # All indices must be in [0, T)
    assert (indices >= 0).all() and (indices < T).all(), "Indices out of range [0, T)"


# ---------------------------------------------------------------------------
# Test 14: scorer normalized sums to ~1
# ---------------------------------------------------------------------------


def test_scorer_normalized_sums(tiny_model, input_ids):
    cfg = TokenImportanceConfig(method="gradient", normalize=True)
    scorer = TokenImportanceScorer(tiny_model, cfg)
    scores = scorer.score(input_ids, target_position=2, target_token=5)
    row_sums = scores.sum(dim=-1)
    for i, s in enumerate(row_sums):
        assert abs(s.item() - 1.0) < 1e-4, f"Row {i} sum = {s.item()}, expected ~1.0"


# ---------------------------------------------------------------------------
# Test 15: scorer score_sequence shape
# ---------------------------------------------------------------------------


def test_scorer_score_sequence_shape(tiny_model, input_ids):
    cfg = TokenImportanceConfig(method="gradient", normalize=True)
    scorer = TokenImportanceScorer(tiny_model, cfg)
    seq_scores = scorer.score_sequence(input_ids)
    assert seq_scores.shape == (B, T), f"Expected ({B}, {T}), got {seq_scores.shape}"
