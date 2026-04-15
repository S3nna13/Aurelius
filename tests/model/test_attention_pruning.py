"""Tests for src/model/attention_pruning.py."""

from __future__ import annotations

import torch
import pytest

from src.model.attention_pruning import (
    PruningConfig,
    compute_head_importance_magnitude,
    compute_head_importance_random,
    get_heads_to_prune,
    create_head_mask,
    apply_head_mask,
    HeadPruner,
    compute_sparsity_ratio,
    PrunedMultiHeadAttention,
)

# ---------------------------------------------------------------------------
# Shared tiny test dimensions
# ---------------------------------------------------------------------------
N_HEADS = 4
D_HEAD = 8
D_MODEL = 16   # N_HEADS * D_HEAD
B = 2
T = 6
PRUNE_RATIO = 0.5

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. PruningConfig — default field values
# ---------------------------------------------------------------------------
def test_pruning_config_defaults():
    cfg = PruningConfig()
    assert cfg.n_heads == 8
    assert cfg.d_head == 64
    assert cfg.prune_ratio == 0.5
    assert cfg.importance_metric == "magnitude"


# ---------------------------------------------------------------------------
# 2. compute_head_importance_magnitude — output shape is (N_HEADS,)
# ---------------------------------------------------------------------------
def test_compute_head_importance_magnitude_shape():
    weight = torch.randn(N_HEADS * D_HEAD, D_MODEL)
    importance = compute_head_importance_magnitude(weight, N_HEADS, D_HEAD)
    assert importance.shape == (N_HEADS,), (
        f"Expected ({N_HEADS},), got {importance.shape}"
    )


# ---------------------------------------------------------------------------
# 3. compute_head_importance_magnitude — all scores are non-negative
# ---------------------------------------------------------------------------
def test_compute_head_importance_magnitude_nonnegative():
    weight = torch.randn(N_HEADS * D_HEAD, D_MODEL)
    importance = compute_head_importance_magnitude(weight, N_HEADS, D_HEAD)
    assert (importance >= 0).all(), "Magnitude importances must be non-negative (L2 norm)"


# ---------------------------------------------------------------------------
# 4. compute_head_importance_random — output shape is (N_HEADS,)
# ---------------------------------------------------------------------------
def test_compute_head_importance_random_shape():
    importance = compute_head_importance_random(N_HEADS, seed=7)
    assert importance.shape == (N_HEADS,), (
        f"Expected ({N_HEADS},), got {importance.shape}"
    )


# ---------------------------------------------------------------------------
# 5. compute_head_importance_random — deterministic with same seed
# ---------------------------------------------------------------------------
def test_compute_head_importance_random_deterministic():
    imp_a = compute_head_importance_random(N_HEADS, seed=99)
    imp_b = compute_head_importance_random(N_HEADS, seed=99)
    assert torch.allclose(imp_a, imp_b), (
        "Random importances with the same seed must be identical"
    )


# ---------------------------------------------------------------------------
# 6. get_heads_to_prune — returned count matches floor(prune_ratio * n_heads)
# ---------------------------------------------------------------------------
def test_get_heads_to_prune_count():
    import math
    importance = torch.tensor([0.1, 0.9, 0.3, 0.7], dtype=torch.float32)
    heads = get_heads_to_prune(importance, PRUNE_RATIO)
    expected = math.floor(PRUNE_RATIO * N_HEADS)
    assert len(heads) == expected, (
        f"Expected {expected} heads to prune, got {len(heads)}"
    )


# ---------------------------------------------------------------------------
# 7. get_heads_to_prune — all returned indices are valid head indices
# ---------------------------------------------------------------------------
def test_get_heads_to_prune_valid_indices():
    importance = torch.rand(N_HEADS)
    heads = get_heads_to_prune(importance, PRUNE_RATIO)
    for idx in heads:
        assert 0 <= idx < N_HEADS, f"Head index {idx} out of range [0, {N_HEADS})"


# ---------------------------------------------------------------------------
# 8. create_head_mask — correct number of True (keep) values
# ---------------------------------------------------------------------------
def test_create_head_mask_true_count():
    import math
    heads_to_prune = [0, 1]   # prune 2 of 4
    mask = create_head_mask(N_HEADS, heads_to_prune)
    assert mask.shape == (N_HEADS,)
    expected_keep = N_HEADS - len(heads_to_prune)
    assert mask.sum().item() == expected_keep, (
        f"Expected {expected_keep} kept heads, got {mask.sum().item()}"
    )


# ---------------------------------------------------------------------------
# 9. apply_head_mask — output preserves input shape
# ---------------------------------------------------------------------------
def test_apply_head_mask_shape():
    attn_output = torch.randn(B, N_HEADS, T, D_HEAD)
    head_mask = torch.ones(N_HEADS, dtype=torch.bool)
    out = apply_head_mask(attn_output, head_mask)
    assert out.shape == attn_output.shape, (
        f"Shape mismatch: expected {attn_output.shape}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 10. apply_head_mask — pruned heads are zeroed out
# ---------------------------------------------------------------------------
def test_apply_head_mask_zeros_pruned_heads():
    attn_output = torch.randn(B, N_HEADS, T, D_HEAD)
    # Prune heads 1 and 3
    head_mask = torch.tensor([True, False, True, False])
    out = apply_head_mask(attn_output, head_mask)

    # Pruned heads should be all zeros
    assert out[:, 1, :, :].abs().sum().item() == pytest.approx(0.0), \
        "Head 1 should be zeroed"
    assert out[:, 3, :, :].abs().sum().item() == pytest.approx(0.0), \
        "Head 3 should be zeroed"

    # Kept heads should be unchanged
    assert torch.allclose(out[:, 0, :, :], attn_output[:, 0, :, :]), \
        "Head 0 should be unchanged"
    assert torch.allclose(out[:, 2, :, :], attn_output[:, 2, :, :]), \
        "Head 2 should be unchanged"


# ---------------------------------------------------------------------------
# 11. HeadPruner.analyze_heads — output shape is (N_HEADS,)
# ---------------------------------------------------------------------------
def test_head_pruner_analyze_heads_shape():
    cfg = PruningConfig(n_heads=N_HEADS, d_head=D_HEAD, prune_ratio=PRUNE_RATIO,
                        importance_metric="magnitude")
    pruner = HeadPruner(cfg)
    weight = torch.randn(N_HEADS * D_HEAD, D_MODEL)
    importance = pruner.analyze_heads(weight)
    assert importance.shape == (N_HEADS,), (
        f"Expected ({N_HEADS},), got {importance.shape}"
    )


# ---------------------------------------------------------------------------
# 12. compute_sparsity_ratio — equals fraction of pruned heads
# ---------------------------------------------------------------------------
def test_compute_sparsity_ratio():
    # 2 pruned out of 4 → sparsity = 0.5
    mask = torch.tensor([True, False, True, False])
    ratio = compute_sparsity_ratio(mask)
    assert ratio == pytest.approx(0.5), f"Expected 0.5, got {ratio}"


# ---------------------------------------------------------------------------
# 13. PrunedMultiHeadAttention — output shape is (B, T, D_MODEL)
# ---------------------------------------------------------------------------
def test_pruned_mha_output_shape():
    torch.manual_seed(1)
    model = PrunedMultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 14. PrunedMultiHeadAttention — masked output differs from unmasked output
# ---------------------------------------------------------------------------
def test_pruned_mha_mask_changes_output():
    torch.manual_seed(2)
    model = PrunedMultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)

    # Full mask — all heads active
    out_full = model(x)

    # Prune half the heads
    pruned_mask = torch.tensor([True, False, True, False])
    model.set_head_mask(pruned_mask)
    out_pruned = model(x)

    assert not torch.allclose(out_full, out_pruned), (
        "Pruned output should differ from unpruned output when heads are masked"
    )
