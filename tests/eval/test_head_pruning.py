"""Tests for src/eval/head_pruning.py -- attention head importance scoring and pruning."""

from __future__ import annotations

import torch
import pytest

from src.eval.head_pruning import (
    HeadPruningConfig,
    HeadMask,
    apply_head_pruning,
    compute_head_entropy,
    evaluate_pruning_impact,
    score_heads_globally,
    select_heads_to_prune,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=4,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)
B, T = 2, 8


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(42)
    m = AureliusTransformer(SMALL_CONFIG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, SMALL_CONFIG.vocab_size, (B, T))


# ---------------------------------------------------------------------------
# 1. HeadPruningConfig defaults
# ---------------------------------------------------------------------------

def test_head_pruning_config_defaults():
    cfg = HeadPruningConfig()
    assert cfg.importance_metric == "gradient"
    assert cfg.prune_fraction == 0.3
    assert cfg.min_heads_per_layer == 1
    assert cfg.global_pruning is True


# ---------------------------------------------------------------------------
# 2. compute_head_entropy output shape
# ---------------------------------------------------------------------------

def test_compute_head_entropy_shape():
    n_heads = 4
    attn = torch.softmax(torch.randn(B, n_heads, T, T), dim=-1)
    entropy = compute_head_entropy(attn)
    assert entropy.shape == (n_heads,), f"Expected ({n_heads},), got {entropy.shape}"


# ---------------------------------------------------------------------------
# 3. compute_head_entropy: uniform attention has maximum entropy
# ---------------------------------------------------------------------------

def test_compute_head_entropy_uniform_is_max():
    n_heads = 4
    uniform = torch.full((B, n_heads, T, T), 1.0 / T)
    uniform_entropy = compute_head_entropy(uniform)

    torch.manual_seed(1)
    peaked = torch.softmax(torch.randn(B, n_heads, T, T) * 10.0, dim=-1)
    peaked_entropy = compute_head_entropy(peaked)

    assert (uniform_entropy >= peaked_entropy - 1e-4).all(), (
        f"Uniform entropy {uniform_entropy} should be >= peaked entropy {peaked_entropy}"
    )


# ---------------------------------------------------------------------------
# 4. compute_head_entropy: focused attention has lower entropy
# ---------------------------------------------------------------------------

def test_compute_head_entropy_focused_is_lower():
    n_heads = 4
    uniform = torch.full((B, n_heads, T, T), 1.0 / T)
    uniform_entropy = compute_head_entropy(uniform)

    focused = torch.zeros(B, n_heads, T, T)
    focused[:, :, :, 0] = 1.0
    focused_entropy = compute_head_entropy(focused)

    assert (uniform_entropy > focused_entropy).all(), (
        "Uniform attention should have higher entropy than focused attention"
    )


# ---------------------------------------------------------------------------
# 5. score_heads_globally returns dict with correct layer keys
# ---------------------------------------------------------------------------

def test_score_heads_globally_returns_layer_keys(model, input_ids):
    cfg = HeadPruningConfig(importance_metric="gradient")
    scores = score_heads_globally(model, input_ids, cfg)
    expected_keys = set(range(SMALL_CONFIG.n_layers))
    assert set(scores.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(scores.keys())}"
    )


# ---------------------------------------------------------------------------
# 6. score_heads_globally scores are non-negative
# ---------------------------------------------------------------------------

def test_score_heads_globally_non_negative(model, input_ids):
    cfg = HeadPruningConfig(importance_metric="gradient")
    scores = score_heads_globally(model, input_ids, cfg)
    for lidx, s in scores.items():
        assert (s >= 0).all(), f"Layer {lidx} has negative scores: {s}"


# ---------------------------------------------------------------------------
# 7. select_heads_to_prune respects prune_fraction
# ---------------------------------------------------------------------------

def test_select_heads_to_prune_respects_fraction():
    n_layers = 2
    n_heads = 4
    torch.manual_seed(7)
    scores = {i: torch.rand(n_heads) for i in range(n_layers)}
    cfg = HeadPruningConfig(prune_fraction=0.5, min_heads_per_layer=1, global_pruning=True)
    heads_to_prune = select_heads_to_prune(scores, cfg)

    total_pruned = sum(len(v) for v in heads_to_prune.values())
    total_heads = n_layers * n_heads
    expected = int(total_heads * cfg.prune_fraction)
    assert total_pruned == expected, (
        f"Expected {expected} heads pruned, got {total_pruned}"
    )


# ---------------------------------------------------------------------------
# 8. select_heads_to_prune respects min_heads_per_layer (never prunes all heads)
# ---------------------------------------------------------------------------

def test_select_heads_to_prune_min_heads_respected():
    n_layers = 2
    n_heads = 4
    scores = {i: torch.zeros(n_heads) for i in range(n_layers)}
    cfg = HeadPruningConfig(prune_fraction=1.0, min_heads_per_layer=1, global_pruning=True)
    heads_to_prune = select_heads_to_prune(scores, cfg)

    for lidx in range(n_layers):
        remaining = n_heads - len(heads_to_prune[lidx])
        assert remaining >= cfg.min_heads_per_layer, (
            f"Layer {lidx}: only {remaining} heads remain, "
            f"min_heads_per_layer={cfg.min_heads_per_layer}"
        )


# ---------------------------------------------------------------------------
# 9. HeadMask apply_mask output shape correct
# ---------------------------------------------------------------------------

def test_head_mask_apply_mask_shape():
    n_layers = 2
    n_heads = 4
    head_dim = 16
    mask_module = HeadMask(n_layers=n_layers, n_heads=n_heads)

    attn_out = torch.randn(B, n_heads, T, head_dim)
    result = mask_module.apply_mask(attn_out, layer_idx=0)
    assert result.shape == attn_out.shape, (
        f"Expected shape {attn_out.shape}, got {result.shape}"
    )


# ---------------------------------------------------------------------------
# 10. HeadMask apply_mask with zeros zeroes output
# ---------------------------------------------------------------------------

def test_head_mask_apply_mask_zeros_output():
    n_layers = 2
    n_heads = 4
    head_dim = 16
    mask_module = HeadMask(n_layers=n_layers, n_heads=n_heads)

    with torch.no_grad():
        mask_module.masks[0] = 0.0

    attn_out = torch.randn(B, n_heads, T, head_dim)
    result = mask_module.apply_mask(attn_out, layer_idx=0)
    assert torch.allclose(result, torch.zeros_like(result)), (
        "Output should be all zeros when mask is zero"
    )


# ---------------------------------------------------------------------------
# 11. apply_head_pruning modifies model (pruned head weights become zero)
# ---------------------------------------------------------------------------

def test_apply_head_pruning_zeros_weights():
    torch.manual_seed(42)
    m = AureliusTransformer(SMALL_CONFIG)
    heads_to_prune = {0: [0], 1: []}
    head_dim = SMALL_CONFIG.head_dim

    o_proj_before = m.layers[0].attn.o_proj.weight.data.clone()
    assert not torch.allclose(
        o_proj_before[:, :head_dim],
        torch.zeros_like(o_proj_before[:, :head_dim])
    ), "Weights should be non-zero before pruning"

    apply_head_pruning(m, heads_to_prune)

    o_proj_after = m.layers[0].attn.o_proj.weight.data
    assert torch.allclose(
        o_proj_after[:, :head_dim],
        torch.zeros_like(o_proj_after[:, :head_dim])
    ), "Pruned head columns should be zeroed out"

    assert torch.allclose(
        o_proj_after[:, head_dim:],
        o_proj_before[:, head_dim:]
    ), "Unpruned head columns should remain unchanged"


# ---------------------------------------------------------------------------
# 12. evaluate_pruning_impact returns correct keys
# ---------------------------------------------------------------------------

def test_evaluate_pruning_impact_keys():
    torch.manual_seed(42)
    m = AureliusTransformer(SMALL_CONFIG)
    ids = torch.randint(0, SMALL_CONFIG.vocab_size, (B, T))
    heads_to_prune = {0: [0], 1: []}
    result = evaluate_pruning_impact(m, ids, heads_to_prune)

    assert "perplexity_before" in result
    assert "perplexity_after" in result
    assert "n_heads_pruned" in result


# ---------------------------------------------------------------------------
# 13. evaluate_pruning_impact n_heads_pruned matches expected
# ---------------------------------------------------------------------------

def test_evaluate_pruning_impact_n_heads_pruned():
    torch.manual_seed(42)
    m = AureliusTransformer(SMALL_CONFIG)
    ids = torch.randint(0, SMALL_CONFIG.vocab_size, (B, T))
    heads_to_prune = {0: [0, 1], 1: [2]}
    result = evaluate_pruning_impact(m, ids, heads_to_prune)

    expected = 3  # 2 from layer 0, 1 from layer 1
    assert result["n_heads_pruned"] == float(expected), (
        f"Expected n_heads_pruned={expected}, got {result['n_heads_pruned']}"
    )
