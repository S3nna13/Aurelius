"""Tests for src/interpretability/token_attribution.py

Uses a small AureliusConfig so the model fits comfortably in CPU memory:
  n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
  d_ff=128, vocab_size=256, max_seq_len=64
"""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.interpretability.token_attribution import (
    AttributionConfig,
    TokenAttribution,
    top_k_tokens,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEQ_LEN = 8
TARGET_POS = SEQ_LEN - 1   # last token position
TARGET_TOK = 5              # arbitrary vocab index


@pytest.fixture(scope="module")
def tiny_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture(scope="module")
def model(tiny_config) -> AureliusTransformer:
    m = AureliusTransformer(tiny_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids(tiny_config) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, tiny_config.vocab_size, (1, SEQ_LEN))


@pytest.fixture(scope="module")
def attrib(model) -> TokenAttribution:
    return TokenAttribution(model, method="gradient")


# ---------------------------------------------------------------------------
# Test 1: AttributionConfig defaults
# ---------------------------------------------------------------------------

def test_attribution_config_defaults():
    cfg = AttributionConfig()
    assert cfg.method == "gradient"
    assert cfg.n_steps == 20
    assert cfg.normalize is True
    assert cfg.baseline_type == "zero"


# ---------------------------------------------------------------------------
# Test 2: gradient_attribution returns (seq_len,) tensor
# ---------------------------------------------------------------------------

def test_gradient_attribution_shape(attrib, input_ids):
    scores = attrib.gradient_attribution(input_ids, TARGET_POS, TARGET_TOK)
    assert scores.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 3: gradient_attribution scores are non-negative
# ---------------------------------------------------------------------------

def test_gradient_attribution_non_negative(attrib, input_ids):
    scores = attrib.gradient_attribution(input_ids, TARGET_POS, TARGET_TOK)
    assert (scores >= 0).all(), "Gradient attribution scores must be non-negative"


# ---------------------------------------------------------------------------
# Test 4: integrated_gradients returns (seq_len,) tensor
# ---------------------------------------------------------------------------

def test_integrated_gradients_shape(attrib, input_ids):
    scores = attrib.integrated_gradients(input_ids, TARGET_POS, TARGET_TOK, n_steps=5)
    assert scores.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 5: attention_rollout returns (seq_len, seq_len) matrix
# ---------------------------------------------------------------------------

def test_attention_rollout_shape(attrib, input_ids):
    rollout = attrib.attention_rollout(input_ids)
    assert rollout.shape == (SEQ_LEN, SEQ_LEN), (
        f"Expected ({SEQ_LEN}, {SEQ_LEN}), got {rollout.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: attention_rollout rows sum to approximately 1.0
# ---------------------------------------------------------------------------

def test_attention_rollout_rows_sum_to_one(attrib, input_ids):
    rollout = attrib.attention_rollout(input_ids)
    row_sums = rollout.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(SEQ_LEN), atol=1e-4), (
        f"Attention rollout rows should sum to 1; got {row_sums}"
    )


# ---------------------------------------------------------------------------
# Test 7: erasure_attribution returns (seq_len,) tensor
# ---------------------------------------------------------------------------

def test_erasure_attribution_shape(attrib, input_ids):
    scores = attrib.erasure_attribution(input_ids, TARGET_POS, TARGET_TOK)
    assert scores.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 8: normalize_attributions returns values in [0, 1]
# ---------------------------------------------------------------------------

def test_normalize_attributions_range(attrib):
    raw = torch.tensor([0.1, 0.5, 1.2, 0.0, 0.8])
    normed = attrib.normalize_attributions(raw)
    assert normed.min().item() >= 0.0 - 1e-6, "Normalized min should be >= 0"
    assert normed.max().item() <= 1.0 + 1e-6, "Normalized max should be <= 1"


# ---------------------------------------------------------------------------
# Test 9: top_k_tokens returns k results
# ---------------------------------------------------------------------------

def test_top_k_tokens_count(input_ids):
    k = 3
    scores = torch.rand(SEQ_LEN)
    result = top_k_tokens(scores, k, input_ids)
    assert len(result) == k, f"Expected {k} results, got {len(result)}"


def test_top_k_tokens_structure(input_ids):
    k = 3
    scores = torch.rand(SEQ_LEN)
    result = top_k_tokens(scores, k, input_ids)
    for token_id, pos, score in result:
        assert isinstance(token_id, int)
        assert isinstance(pos, int)
        assert isinstance(score, float)
        assert 0 <= pos < SEQ_LEN


# ---------------------------------------------------------------------------
# Test 10: Different methods give different attributions
# ---------------------------------------------------------------------------

def test_different_methods_differ(model, input_ids):
    """Gradient, integrated_gradients, and erasure should not all produce identical scores."""
    attrib_obj = TokenAttribution(model, method="gradient")

    grad_scores = attrib_obj.gradient_attribution(input_ids, TARGET_POS, TARGET_TOK)
    ig_scores = attrib_obj.integrated_gradients(input_ids, TARGET_POS, TARGET_TOK, n_steps=3)
    erasure_scores = attrib_obj.erasure_attribution(input_ids, TARGET_POS, TARGET_TOK)

    # At least one pair must differ (not all three identical)
    grad_eq_ig = torch.allclose(grad_scores, ig_scores, atol=1e-4)
    grad_eq_erasure = torch.allclose(grad_scores, erasure_scores, atol=1e-4)

    assert not (grad_eq_ig and grad_eq_erasure), (
        "All three methods produced identical attributions — "
        "this is extremely unlikely and suggests a bug"
    )
