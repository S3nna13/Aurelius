"""Tests for position_interpolation.py — 16 test cases covering all functions."""

from __future__ import annotations

import math
import pytest
import torch

from src.model.position_interpolation import (
    PositionInterpolationConfig,
    linear_interpolate_positions,
    ntk_rope_base,
    build_rope_freqs,
    apply_rope,
    ContextLengthExtender,
    dynamic_ntk_scale,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helper: small test model
# ---------------------------------------------------------------------------

def make_tiny_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# 1. PositionInterpolationConfig defaults
# ---------------------------------------------------------------------------

def test_position_interpolation_config_defaults():
    cfg = PositionInterpolationConfig()
    assert cfg.original_max_len == 8192
    assert cfg.target_max_len == 32768
    assert cfg.method == "linear"
    assert cfg.ntk_alpha == 1.0


# ---------------------------------------------------------------------------
# 2. linear_interpolate_positions output shape is (seq_len,)
# ---------------------------------------------------------------------------

def test_linear_interpolate_positions_shape():
    seq_len = 128
    result = linear_interpolate_positions(seq_len, original_max_len=512, target_max_len=2048)
    assert result.shape == (seq_len,)


# ---------------------------------------------------------------------------
# 3. linear_interpolate_positions positions are scaled (< original_max_len)
# ---------------------------------------------------------------------------

def test_linear_interpolate_positions_scaled():
    seq_len = 100
    original_max_len = 512
    target_max_len = 2048
    result = linear_interpolate_positions(seq_len, original_max_len, target_max_len)
    assert result.max().item() < original_max_len


# ---------------------------------------------------------------------------
# 4. linear_interpolate_positions first position is 0
# ---------------------------------------------------------------------------

def test_linear_interpolate_positions_first_is_zero():
    result = linear_interpolate_positions(64, original_max_len=1024, target_max_len=4096)
    assert result[0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. linear_interpolate_positions with scale=1.0: returns integers
# ---------------------------------------------------------------------------

def test_linear_interpolate_positions_scale_one():
    seq_len = 32
    result = linear_interpolate_positions(seq_len, original_max_len=512, target_max_len=512)
    expected = torch.arange(seq_len, dtype=torch.float32)
    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. ntk_rope_base returns float > original base
# ---------------------------------------------------------------------------

def test_ntk_rope_base_larger_than_original():
    base = 10000.0
    new_base = ntk_rope_base(base, head_dim=64, original_max_len=512, target_max_len=2048, alpha=1.0)
    assert isinstance(new_base, float)
    assert new_base > base


# ---------------------------------------------------------------------------
# 7. ntk_rope_base with alpha=1.0 and scale=1.0: returns original base
# ---------------------------------------------------------------------------

def test_ntk_rope_base_no_scaling():
    base = 10000.0
    new_base = ntk_rope_base(base, head_dim=64, original_max_len=1024, target_max_len=1024, alpha=1.0)
    assert new_base == pytest.approx(base, rel=1e-5)


# ---------------------------------------------------------------------------
# 8. build_rope_freqs output shape is (seq_len, head_dim)
# ---------------------------------------------------------------------------

def test_build_rope_freqs_shape():
    seq_len = 64
    head_dim = 32
    freqs = build_rope_freqs(head_dim, seq_len)
    assert freqs.shape == (seq_len, head_dim)


# ---------------------------------------------------------------------------
# 9. build_rope_freqs with custom positions
# ---------------------------------------------------------------------------

def test_build_rope_freqs_custom_positions():
    seq_len = 16
    head_dim = 32
    positions = torch.linspace(0, 5.0, seq_len)
    freqs = build_rope_freqs(head_dim, seq_len, positions=positions)
    assert freqs.shape == (seq_len, head_dim)
    # First row: positions[0]=0 => cos(0)=1, sin(0)=0 for all dims
    assert freqs[0, 0::2].allclose(torch.ones(head_dim // 2), atol=1e-6)
    assert freqs[0, 1::2].allclose(torch.zeros(head_dim // 2), atol=1e-6)


# ---------------------------------------------------------------------------
# 10. apply_rope output shape matches input (B, H, T, head_dim)
# ---------------------------------------------------------------------------

def test_apply_rope_output_shape():
    B, H, T, D = 2, 4, 16, 32
    x = torch.randn(B, H, T, D)
    freqs = build_rope_freqs(D, T)
    out = apply_rope(x, freqs)
    assert out.shape == (B, H, T, D)


# ---------------------------------------------------------------------------
# 11. apply_rope with identity freqs (all zeros): output == input
# ---------------------------------------------------------------------------

def test_apply_rope_identity():
    B, H, T, D = 2, 2, 8, 16
    x = torch.randn(B, H, T, D)
    # cos=1, sin=0 => identity rotation
    freqs = torch.zeros(T, D)
    freqs[:, 0::2] = 1.0  # cos component = 1
    out = apply_rope(x, freqs)
    assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. ContextLengthExtender.patch_rope changes model.freqs_cis shape
# ---------------------------------------------------------------------------

def test_context_length_extender_patch_rope_changes_shape():
    model = make_tiny_model()
    original_shape = model.freqs_cis.shape  # (512, head_dim//2)

    cfg = PositionInterpolationConfig(
        original_max_len=512,
        target_max_len=1024,
        method="linear",
    )
    extender = ContextLengthExtender(model, cfg)
    extender.patch_rope()

    new_shape = model.freqs_cis.shape
    assert new_shape[0] == 1024, f"Expected 1024 positions, got {new_shape[0]}"
    assert new_shape[0] != original_shape[0]

    extender.restore_rope()


# ---------------------------------------------------------------------------
# 13. ContextLengthExtender.restore_rope restores original shape
# ---------------------------------------------------------------------------

def test_context_length_extender_restore_rope():
    model = make_tiny_model()
    original_shape = model.freqs_cis.shape

    cfg = PositionInterpolationConfig(
        original_max_len=512,
        target_max_len=1024,
        method="ntk",
    )
    extender = ContextLengthExtender(model, cfg)
    extender.patch_rope()
    extender.restore_rope()

    assert model.freqs_cis.shape == original_shape


# ---------------------------------------------------------------------------
# 14. ContextLengthExtender.generate_extended returns logits
# ---------------------------------------------------------------------------

def test_context_length_extender_generate_extended():
    model = make_tiny_model()
    model.eval()

    cfg = PositionInterpolationConfig(
        original_max_len=512,
        target_max_len=1024,
        method="linear",
    )
    extender = ContextLengthExtender(model, cfg)

    B, T = 1, 8
    input_ids = torch.randint(0, 256, (B, T))
    logits = extender.generate_extended(input_ids)

    assert logits.shape == (B, T, 256)


# ---------------------------------------------------------------------------
# 15. dynamic_ntk_scale returns 1.0 when seq_len <= original_max_len
# ---------------------------------------------------------------------------

def test_dynamic_ntk_scale_no_scaling():
    scale = dynamic_ntk_scale(seq_len=512, original_max_len=512, head_dim=64)
    assert scale == pytest.approx(1.0)

    scale2 = dynamic_ntk_scale(seq_len=256, original_max_len=512, head_dim=64)
    assert scale2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 16. dynamic_ntk_scale > 1.0 when seq_len > original_max_len
# ---------------------------------------------------------------------------

def test_dynamic_ntk_scale_greater_than_one():
    scale = dynamic_ntk_scale(seq_len=2048, original_max_len=512, head_dim=64)
    assert scale > 1.0
