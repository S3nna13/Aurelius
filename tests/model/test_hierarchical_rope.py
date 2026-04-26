"""Tests for hierarchical_rope.py — Hierarchical Rotary Position Embeddings."""

import pytest
import torch

from src.model.hierarchical_rope import (
    HierarchicalRoPEAttention,
    HierarchicalRoPEConfig,
    RoPEScaleScheduler,
    apply_hierarchical_rope,
    apply_rope_single_scale,
    compute_position_bias,
    compute_rope_frequencies,
    interpolate_rope_frequencies,
)

# Common test dimensions
B, H, T, HEAD_DIM = 2, 2, 8, 32
D_MODEL = 64  # = H * HEAD_DIM


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = HierarchicalRoPEConfig()
    assert cfg.n_scales == 3
    assert len(cfg.scale_factors) == 3


# ---------------------------------------------------------------------------
# 2. compute_rope_frequencies shape
# ---------------------------------------------------------------------------
def test_compute_rope_frequencies_shape():
    freqs = compute_rope_frequencies(HEAD_DIM)
    assert freqs.shape == (HEAD_DIM // 2,)


# ---------------------------------------------------------------------------
# 3. compute_rope_frequencies: larger scale → smaller frequencies
# ---------------------------------------------------------------------------
def test_compute_rope_frequencies_scale():
    freqs_base = compute_rope_frequencies(HEAD_DIM, scale=1.0)
    freqs_large = compute_rope_frequencies(HEAD_DIM, scale=4.0)
    # larger scale → denominator larger → frequencies smaller
    # skip index 0 where exponent is 0 and both equal 1.0
    assert (freqs_large[1:] < freqs_base[1:]).all()


# ---------------------------------------------------------------------------
# 4. apply_rope_single_scale shape
# ---------------------------------------------------------------------------
def test_apply_rope_single_scale_shape():
    x = torch.randn(B, H, T, HEAD_DIM)
    freqs = compute_rope_frequencies(HEAD_DIM)
    positions = torch.arange(T, dtype=torch.float32)
    out = apply_rope_single_scale(x, freqs, positions)
    assert out.shape == (B, H, T, HEAD_DIM)


# ---------------------------------------------------------------------------
# 5. apply_rope_single_scale: rotation preserves vector norms
# ---------------------------------------------------------------------------
def test_apply_rope_single_scale_norm_preserved():
    x = torch.randn(B, H, T, HEAD_DIM)
    freqs = compute_rope_frequencies(HEAD_DIM)
    positions = torch.arange(T, dtype=torch.float32)
    out = apply_rope_single_scale(x, freqs, positions)

    orig_norms = x.norm(dim=-1)
    rot_norms = out.norm(dim=-1)
    assert torch.allclose(orig_norms, rot_norms, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. apply_rope_single_scale: different positions → different outputs
# ---------------------------------------------------------------------------
def test_apply_rope_shifts_with_position():
    x = torch.randn(B, H, T, HEAD_DIM)
    freqs = compute_rope_frequencies(HEAD_DIM)

    pos_a = torch.arange(T, dtype=torch.float32)
    pos_b = torch.arange(T, dtype=torch.float32) + 100.0

    out_a = apply_rope_single_scale(x, freqs, pos_a)
    out_b = apply_rope_single_scale(x, freqs, pos_b)
    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# 7. apply_hierarchical_rope shape
# ---------------------------------------------------------------------------
def test_apply_hierarchical_rope_shape():
    cfg = HierarchicalRoPEConfig(head_dim=HEAD_DIM)
    x = torch.randn(B, H, T, HEAD_DIM)
    out = apply_hierarchical_rope(x, cfg)
    assert out.shape == (B, H, T, HEAD_DIM)


# ---------------------------------------------------------------------------
# 8. apply_hierarchical_rope ≠ single scale (with n_scales > 1)
# ---------------------------------------------------------------------------
def test_apply_hierarchical_rope_different_from_single():
    cfg = HierarchicalRoPEConfig(head_dim=HEAD_DIM, n_scales=3, scale_factors=[1.0, 4.0, 16.0])
    x = torch.randn(B, H, T, HEAD_DIM)

    hierarchical_out = apply_hierarchical_rope(x, cfg)

    # single-scale at scale=1.0
    freqs = compute_rope_frequencies(HEAD_DIM, scale=1.0)
    positions = torch.arange(T, dtype=torch.float32)
    single_out = apply_rope_single_scale(x, freqs, positions)

    assert not torch.allclose(hierarchical_out, single_out)


# ---------------------------------------------------------------------------
# 9. compute_position_bias shape
# ---------------------------------------------------------------------------
def test_compute_position_bias_shape():
    bias = compute_position_bias(T, n_scales=3, scale_factors=[1.0, 4.0, 16.0])
    assert bias.shape == (T, T)


# ---------------------------------------------------------------------------
# 10. compute_position_bias: symmetric
# ---------------------------------------------------------------------------
def test_compute_position_bias_symmetric():
    bias = compute_position_bias(T, n_scales=3, scale_factors=[1.0, 4.0, 16.0])
    assert torch.allclose(bias, bias.T, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. HierarchicalRoPEAttention output shape
# ---------------------------------------------------------------------------
def test_hierarchical_rope_attention_shape():
    cfg = HierarchicalRoPEConfig(head_dim=HEAD_DIM)
    model = HierarchicalRoPEAttention(d_model=D_MODEL, n_heads=H, cfg=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 12. HierarchicalRoPEAttention: works with custom positions
# ---------------------------------------------------------------------------
def test_hierarchical_rope_attention_custom_positions():
    cfg = HierarchicalRoPEConfig(head_dim=HEAD_DIM)
    model = HierarchicalRoPEAttention(d_model=D_MODEL, n_heads=H, cfg=cfg)
    x = torch.randn(B, T, D_MODEL)
    positions = torch.arange(T, dtype=torch.float32) * 2.0  # stride-2 positions
    out = model(x, positions=positions)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 13. interpolate_rope_frequencies: target_scale=2x → halved frequencies
# ---------------------------------------------------------------------------
def test_interpolate_rope_frequencies():
    freqs = compute_rope_frequencies(HEAD_DIM)
    scaled = interpolate_rope_frequencies(freqs, target_scale=2.0, current_scale=1.0)
    assert torch.allclose(scaled, freqs * 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# 14. RoPEScaleScheduler: get_scales(0) returns init_scales
# ---------------------------------------------------------------------------
def test_rope_scale_scheduler_init():
    init = [1.0, 2.0, 4.0]
    target = [2.0, 8.0, 32.0]
    scheduler = RoPEScaleScheduler(init, target, warmup_steps=100)
    result = scheduler.get_scales(0)
    assert result == pytest.approx(init)


# ---------------------------------------------------------------------------
# 15. RoPEScaleScheduler: get_scales(warmup_steps) returns target_scales
# ---------------------------------------------------------------------------
def test_rope_scale_scheduler_target():
    init = [1.0, 2.0, 4.0]
    target = [2.0, 8.0, 32.0]
    warmup = 100
    scheduler = RoPEScaleScheduler(init, target, warmup_steps=warmup)
    result = scheduler.get_scales(warmup)
    assert result == pytest.approx(target)


# ---------------------------------------------------------------------------
# 16. RoPEScaleScheduler: midpoint is average of init and target
# ---------------------------------------------------------------------------
def test_rope_scale_scheduler_midpoint():
    init = [1.0, 2.0, 4.0]
    target = [3.0, 10.0, 20.0]
    warmup = 100
    scheduler = RoPEScaleScheduler(init, target, warmup_steps=warmup)
    result = scheduler.get_scales(50)
    expected = [(i + t) / 2 for i, t in zip(init, target)]
    assert result == pytest.approx(expected, rel=1e-5)
