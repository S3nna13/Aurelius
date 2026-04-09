"""Tests for src/eval/attention_analysis.py."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.attention_analysis import (
    AttentionAnalysisConfig,
    attention_flow,
    attention_rollout,
    compute_attention_entropy,
    compute_head_importance,
    detect_redundant_heads,
    extract_attention_maps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def make_random_attn(B: int, H: int, T: int, n_layers: int) -> list[torch.Tensor]:
    """Create random softmax attention matrices for testing."""
    matrices = []
    for _ in range(n_layers):
        raw = torch.rand(B, H, T, T)
        softmaxed = torch.softmax(raw, dim=-1)
        matrices.append(softmaxed)
    return matrices


# ---------------------------------------------------------------------------
# Test AttentionAnalysisConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = AttentionAnalysisConfig()
    assert cfg.n_heads == 8
    assert cfg.n_layers == 12
    assert cfg.rollout_discard_ratio == 0.0


def test_config_custom():
    cfg = AttentionAnalysisConfig(n_heads=4, n_layers=6, rollout_discard_ratio=0.1)
    assert cfg.n_heads == 4
    assert cfg.n_layers == 6
    assert cfg.rollout_discard_ratio == 0.1


# ---------------------------------------------------------------------------
# Test compute_attention_entropy
# ---------------------------------------------------------------------------

def test_entropy_shape():
    B, H, T = 2, 4, 10
    attn = torch.softmax(torch.rand(B, H, T, T), dim=-1)
    entropy = compute_attention_entropy(attn)
    assert entropy.shape == (B, H, T)


def test_entropy_uniform():
    """Uniform attention over T positions should give entropy = log(T)."""
    B, H, T = 1, 2, 8
    attn = torch.full((B, H, T, T), 1.0 / T)
    entropy = compute_attention_entropy(attn)
    expected = math.log(T)
    assert torch.allclose(entropy, torch.full_like(entropy, expected), atol=1e-4)


def test_entropy_nonnegative():
    B, H, T = 2, 3, 6
    attn = torch.softmax(torch.rand(B, H, T, T), dim=-1)
    entropy = compute_attention_entropy(attn)
    assert (entropy >= 0).all()


# ---------------------------------------------------------------------------
# Test attention_rollout
# ---------------------------------------------------------------------------

def test_rollout_shape():
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    rollout = attention_rollout(matrices)
    assert rollout.shape == (B, T, T)


def test_rollout_single_layer():
    """Single-layer rollout = 0.5*avg_head + 0.5*I."""
    B, H, T = 1, 2, 4
    attn = torch.softmax(torch.rand(B, H, T, T), dim=-1)
    rollout = attention_rollout([attn])
    avg = attn.mean(dim=1)  # (B, T, T)
    eye = torch.eye(T).unsqueeze(0).expand(B, -1, -1)
    expected = 0.5 * avg + 0.5 * eye
    assert torch.allclose(rollout, expected, atol=1e-5)


def test_rollout_with_discard_ratio():
    """Rollout with discard_ratio=0.5 should still produce correct shape."""
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    rollout = attention_rollout(matrices, discard_ratio=0.5)
    assert rollout.shape == (B, T, T)


def test_rollout_empty_raises():
    with pytest.raises(ValueError):
        attention_rollout([])


# ---------------------------------------------------------------------------
# Test compute_head_importance
# ---------------------------------------------------------------------------

def test_head_importance_shape():
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    importance = compute_head_importance(matrices)
    assert importance.shape == (L, H)


def test_head_importance_nonnegative():
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    importance = compute_head_importance(matrices)
    assert (importance >= 0).all()


# ---------------------------------------------------------------------------
# Test detect_redundant_heads
# ---------------------------------------------------------------------------

def test_redundant_heads_returns_list_of_tuples():
    L, H = 3, 4
    importance = torch.rand(L, H)
    redundant = detect_redundant_heads(importance, threshold=0.5)
    assert isinstance(redundant, list)
    for item in redundant:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_redundant_heads_threshold_one_all_redundant():
    """threshold=1.0 should mark all heads as redundant (importance is always < 1)."""
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    importance = compute_head_importance(matrices)
    # All importance values should be < 1.0 for softmax attention
    redundant = detect_redundant_heads(importance, threshold=1.0)
    assert len(redundant) == L * H


def test_redundant_heads_threshold_zero_none_redundant():
    """threshold=0.0 should return no redundant heads (all importance >= 0)."""
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    importance = compute_head_importance(matrices)
    redundant = detect_redundant_heads(importance, threshold=0.0)
    assert len(redundant) == 0


# ---------------------------------------------------------------------------
# Test attention_flow
# ---------------------------------------------------------------------------

def test_attention_flow_shape():
    B, H, T, L = 2, 4, 8, 3
    matrices = make_random_attn(B, H, T, L)
    flow = attention_flow(matrices)
    assert flow.shape == (B, T, T)


def test_attention_flow_empty_raises():
    with pytest.raises(ValueError):
        attention_flow([])


def test_attention_flow_single_layer():
    """Single-layer flow should equal the head-averaged attention."""
    B, H, T = 1, 2, 4
    attn = torch.softmax(torch.rand(B, H, T, T), dim=-1)
    flow = attention_flow([attn])
    expected = attn.mean(dim=1)
    assert torch.allclose(flow, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test extract_attention_maps
# ---------------------------------------------------------------------------

def test_extract_attention_maps_returns_dict(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    result = extract_attention_maps(small_model, input_ids, layer_indices=[0, 1])
    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result


def test_extract_attention_maps_correct_keys(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    result = extract_attention_maps(small_model, input_ids, layer_indices=[0])
    assert set(result.keys()) == {0}


def test_extract_attention_maps_tensor_shape(small_model):
    """Each captured tensor should have shape (B, H, T, T)."""
    B, T = 1, 8
    input_ids = torch.randint(0, 256, (B, T))
    result = extract_attention_maps(small_model, input_ids, layer_indices=[0, 1])
    for layer_idx, attn in result.items():
        assert attn.shape == (B, 2, T, T), (
            f"Layer {layer_idx}: expected ({B}, 2, {T}, {T}), got {attn.shape}"
        )


def test_extract_attention_maps_hooks_removed(small_model):
    """Hooks should be removed after extract_attention_maps returns."""
    input_ids = torch.randint(0, 256, (1, 4))
    # Run once to capture
    result1 = extract_attention_maps(small_model, input_ids, layer_indices=[0])
    # Run again — should not fail and not accumulate state inside the function
    result2 = extract_attention_maps(small_model, input_ids, layer_indices=[0])
    # Both results should be valid tensors with the same shape
    assert result1[0].shape == result2[0].shape
