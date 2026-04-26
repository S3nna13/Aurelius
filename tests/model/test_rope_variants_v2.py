"""Tests for rope_variants_v2.py: RoPEConfig, frequency functions, rotation utilities,
apply_rope, RoPEVariant module, and get_rope_freqs dispatcher.

All tests use small dimensions (d_head=8 or 16) and short sequences (T=4..16)
to keep execution fast.
"""

from __future__ import annotations

import pytest
import torch

from src.model.rope_variants_v2 import (
    RoPEConfig,
    RoPEVariant,
    apply_rope,
    build_rotation_matrix,
    compute_linear_scaled_freqs,
    compute_ntk_freqs,
    compute_standard_freqs,
    compute_yarn_freqs,
    get_rope_freqs,
)

torch.manual_seed(42)

# Small dims used across tests
D_HEAD = 16
HALF = D_HEAD // 2
T = 8
B = 2


# ===========================================================================
# 1. RoPEConfig defaults
# ===========================================================================


def test_rope_config_defaults():
    cfg = RoPEConfig()
    assert cfg.d_head == 64
    assert cfg.base == 10000.0
    assert cfg.max_seq_len == 2048
    assert cfg.rope_type == "standard"
    assert cfg.scale_factor == 1.0
    assert cfg.yarn_beta_fast == 32.0
    assert cfg.yarn_beta_slow == 1.0


# ===========================================================================
# 2. RoPEConfig invalid rope_type raises ValueError
# ===========================================================================


def test_rope_config_invalid_type_raises():
    with pytest.raises(ValueError, match="rope_type must be one of"):
        RoPEConfig(rope_type="unknown_variant")


# ===========================================================================
# 3. compute_standard_freqs shape
# ===========================================================================


def test_compute_standard_freqs_shape():
    freqs = compute_standard_freqs(D_HEAD)
    assert freqs.shape == (HALF,), f"Expected ({HALF},), got {freqs.shape}"


# ===========================================================================
# 4. compute_standard_freqs values are positive and strictly decreasing
# ===========================================================================


def test_compute_standard_freqs_positive_decreasing():
    freqs = compute_standard_freqs(D_HEAD, base=10000.0)
    assert (freqs > 0).all(), "All standard freqs must be positive"
    # Strictly decreasing: freqs[i] > freqs[i+1]
    diffs = freqs[:-1] - freqs[1:]
    assert (diffs > 0).all(), "Standard freqs must be strictly decreasing"


# ===========================================================================
# 5. compute_standard_freqs known values
# ===========================================================================


def test_compute_standard_freqs_known_values():
    d = 4  # half = 2
    base = 10000.0
    freqs = compute_standard_freqs(d, base)
    # i=0: 1/10000^0 = 1.0
    # i=1: 1/10000^(2/4) = 1/100 = 0.01
    expected = torch.tensor([1.0, 0.01])
    assert torch.allclose(freqs, expected, atol=1e-6), f"Got {freqs}"


# ===========================================================================
# 6. compute_linear_scaled_freqs are smaller than standard by scale_factor
# ===========================================================================


def test_compute_linear_scaled_freqs_shape_and_ratio():
    scale = 2.0
    freqs_std = compute_standard_freqs(D_HEAD, base=10000.0)
    freqs_lin = compute_linear_scaled_freqs(D_HEAD, base=10000.0, scale_factor=scale)
    assert freqs_lin.shape == (HALF,), f"Expected ({HALF},), got {freqs_lin.shape}"
    ratio = freqs_std / freqs_lin
    assert torch.allclose(ratio, torch.full_like(ratio, scale), atol=1e-5), (
        "Linear scaled freqs should be standard freqs / scale_factor"
    )


# ===========================================================================
# 7. compute_ntk_freqs shape
# ===========================================================================


def test_compute_ntk_freqs_shape():
    freqs = compute_ntk_freqs(D_HEAD, base=10000.0, scale_factor=2.0, max_seq_len=2048)
    assert freqs.shape == (HALF,), f"Expected ({HALF},), got {freqs.shape}"


# ===========================================================================
# 8. compute_ntk_freqs with scale_factor=1.0 matches standard (same base)
# ===========================================================================


def test_compute_ntk_freqs_scale_one_matches_standard():
    freqs_std = compute_standard_freqs(D_HEAD, base=10000.0)
    freqs_ntk = compute_ntk_freqs(D_HEAD, base=10000.0, scale_factor=1.0, max_seq_len=2048)
    # scale_factor=1 -> new_base = base * 1^(...) = base, so should match
    assert torch.allclose(freqs_std, freqs_ntk, atol=1e-5), (
        "NTK freqs with scale_factor=1 should match standard freqs"
    )


# ===========================================================================
# 9. compute_yarn_freqs shape
# ===========================================================================


def test_compute_yarn_freqs_shape():
    freqs = compute_yarn_freqs(
        D_HEAD, base=10000.0, scale_factor=4.0, beta_fast=32.0, beta_slow=1.0
    )
    assert freqs.shape == (HALF,), f"Expected ({HALF},), got {freqs.shape}"


# ===========================================================================
# 10. compute_yarn_freqs values are positive
# ===========================================================================


def test_compute_yarn_freqs_positive():
    freqs = compute_yarn_freqs(
        D_HEAD, base=10000.0, scale_factor=4.0, beta_fast=32.0, beta_slow=1.0
    )
    assert (freqs > 0).all(), "YaRN freqs must all be positive"


# ===========================================================================
# 11. build_rotation_matrix cos/sin shapes
# ===========================================================================


def test_build_rotation_matrix_shapes():
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    assert cos.shape == (T, HALF), f"cos shape: expected ({T}, {HALF}), got {cos.shape}"
    assert sin.shape == (T, HALF), f"sin shape: expected ({T}, {HALF}), got {sin.shape}"


# ===========================================================================
# 12. build_rotation_matrix cos values in [-1, 1]
# ===========================================================================


def test_build_rotation_matrix_cos_range():
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    assert (cos >= -1.0).all() and (cos <= 1.0).all(), "cos values must lie in [-1, 1]"


# ===========================================================================
# 13. build_rotation_matrix first position has cos=1, sin=0
# ===========================================================================


def test_build_rotation_matrix_first_position():
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    # position 0: angle = 0 * freq = 0 for all dims
    assert torch.allclose(cos[0], torch.ones(HALF), atol=1e-6), "cos at pos 0 should be 1"
    assert torch.allclose(sin[0], torch.zeros(HALF), atol=1e-6), "sin at pos 0 should be 0"


# ===========================================================================
# 14. apply_rope output shape matches input
# ===========================================================================


def test_apply_rope_output_shape():
    x = torch.randn(B, T, D_HEAD)
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    out = apply_rope(x, cos, sin)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B}, {T}, {D_HEAD}), got {out.shape}"


# ===========================================================================
# 15. apply_rope output is finite
# ===========================================================================


def test_apply_rope_output_finite():
    x = torch.randn(B, T, D_HEAD)
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    out = apply_rope(x, cos, sin)
    assert torch.isfinite(out).all(), "apply_rope output must be finite"


# ===========================================================================
# 16. apply_rope at position 0 is identity (cos=1, sin=0 -> no rotation)
# ===========================================================================


def test_apply_rope_identity_at_position_zero():
    x = torch.randn(B, 1, D_HEAD)
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, 1)  # only position 0
    out = apply_rope(x, cos, sin)
    # At pos 0: cos=1, sin=0 -> [x1*1 - x2*0, x1*0 + x2*1] = [x1, x2] = x
    assert torch.allclose(out, x, atol=1e-6), "RoPE at position 0 should be identity"


# ===========================================================================
# 17. RoPEVariant.forward shape preserved
# ===========================================================================


def test_rope_variant_forward_shape():
    cfg = RoPEConfig(d_head=D_HEAD, max_seq_len=32)
    model = RoPEVariant(cfg)
    x = torch.randn(B, T, D_HEAD)
    out = model(x)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B}, {T}, {D_HEAD}), got {out.shape}"


# ===========================================================================
# 18. RoPEVariant.forward output is finite
# ===========================================================================


def test_rope_variant_forward_finite():
    cfg = RoPEConfig(d_head=D_HEAD, max_seq_len=32)
    model = RoPEVariant(cfg)
    x = torch.randn(B, T, D_HEAD)
    out = model(x)
    assert torch.isfinite(out).all(), "RoPEVariant forward output must be finite"


# ===========================================================================
# 19. RoPEVariant.extend_to updates buffer to new length
# ===========================================================================


def test_rope_variant_extend_to():
    original_len = 16
    new_len = 64
    cfg = RoPEConfig(d_head=D_HEAD, max_seq_len=original_len)
    model = RoPEVariant(cfg)

    assert model.cos_cached.shape == (original_len, HALF), (
        f"Initial cos buffer should be ({original_len}, {HALF})"
    )

    model.extend_to(new_len)

    assert model.cos_cached.shape == (new_len, HALF), (
        f"After extend_to({new_len}), cos buffer should be ({new_len}, {HALF})"
    )
    assert model.sin_cached.shape == (new_len, HALF), (
        f"After extend_to({new_len}), sin buffer should be ({new_len}, {HALF})"
    )


# ===========================================================================
# 20. RoPEVariant.extend_to allows forward on longer sequence
# ===========================================================================


def test_rope_variant_extend_to_forward_works():
    cfg = RoPEConfig(d_head=D_HEAD, max_seq_len=8)
    model = RoPEVariant(cfg)
    long_T = 32
    model.extend_to(long_T)
    x = torch.randn(B, long_T, D_HEAD)
    out = model(x)
    assert out.shape == (B, long_T, D_HEAD)
    assert torch.isfinite(out).all()


# ===========================================================================
# 21. get_rope_freqs standard type returns correct shape
# ===========================================================================


def test_get_rope_freqs_standard_shape():
    cfg = RoPEConfig(d_head=D_HEAD, rope_type="standard")
    freqs = get_rope_freqs(cfg)
    assert freqs.shape == (HALF,), f"Expected ({HALF},), got {freqs.shape}"


# ===========================================================================
# 22. get_rope_freqs all types return (d_head//2,) shape
# ===========================================================================


@pytest.mark.parametrize("rope_type", ["standard", "linear_scaled", "dynamic_ntk", "yarn"])
def test_get_rope_freqs_all_types_shape(rope_type):
    cfg = RoPEConfig(d_head=D_HEAD, rope_type=rope_type, scale_factor=2.0)
    freqs = get_rope_freqs(cfg)
    assert freqs.shape == (HALF,), f"[{rope_type}] Expected ({HALF},), got {freqs.shape}"


# ===========================================================================
# 23. get_rope_freqs standard matches compute_standard_freqs directly
# ===========================================================================


def test_get_rope_freqs_standard_matches_direct():
    cfg = RoPEConfig(d_head=D_HEAD, base=10000.0, rope_type="standard")
    freqs_via_dispatch = get_rope_freqs(cfg)
    freqs_direct = compute_standard_freqs(D_HEAD, 10000.0)
    assert torch.allclose(freqs_via_dispatch, freqs_direct, atol=1e-7)


# ===========================================================================
# 24. RoPEVariant works for all rope_types
# ===========================================================================


@pytest.mark.parametrize("rope_type", ["standard", "linear_scaled", "dynamic_ntk", "yarn"])
def test_rope_variant_all_types_forward(rope_type):
    cfg = RoPEConfig(d_head=D_HEAD, max_seq_len=32, rope_type=rope_type, scale_factor=2.0)
    model = RoPEVariant(cfg)
    x = torch.randn(B, T, D_HEAD)
    out = model(x)
    assert out.shape == (B, T, D_HEAD), f"[{rope_type}] shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), f"[{rope_type}] non-finite output"


# ===========================================================================
# 25. apply_rope preserves L2 norm per token (rotation is norm-preserving)
# ===========================================================================


def test_apply_rope_norm_preserving():
    x = torch.randn(B, T, D_HEAD)
    freqs = compute_standard_freqs(D_HEAD)
    cos, sin = build_rotation_matrix(freqs, T)
    out = apply_rope(x, cos, sin)

    # Check that ||out[b, t, :]|| ≈ ||x[b, t, :]|| for all b, t
    x_norms = x.norm(dim=-1)
    out_norms = out.norm(dim=-1)
    assert torch.allclose(x_norms, out_norms, atol=1e-5), (
        "RoPE rotation should preserve L2 norm of each token"
    )
