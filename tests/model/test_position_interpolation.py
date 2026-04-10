"""Tests for position_interpolation.py — 14 test cases covering all functions."""

import math
import pytest
import torch

from src.model.position_interpolation import (
    PIConfig,
    apply_rope_with_freqs,
    compute_rope_freqs,
    dynamic_ntk_interpolation,
    linear_position_interpolation,
    ntk_aware_interpolation,
    yarn_interpolation,
    PositionInterpolator,
)


# ---------------------------------------------------------------------------
# 1. PIConfig defaults
# ---------------------------------------------------------------------------
def test_piconfig_defaults():
    cfg = PIConfig()
    assert cfg.method == "linear"
    assert cfg.scale_factor == 4.0
    assert cfg.base_theta == 10000.0
    assert cfg.original_max_len == 2048
    assert cfg.extended_max_len == 8192


# ---------------------------------------------------------------------------
# 2. compute_rope_freqs — shape
# ---------------------------------------------------------------------------
def test_compute_rope_freqs_shape():
    head_dim = 64
    freqs = compute_rope_freqs(head_dim)
    assert freqs.shape == (head_dim // 2,)


# ---------------------------------------------------------------------------
# 3. compute_rope_freqs — values decrease monotonically
# ---------------------------------------------------------------------------
def test_compute_rope_freqs_monotonically_decreasing():
    freqs = compute_rope_freqs(64)
    # Each successive frequency should be <= the previous one
    diffs = freqs[1:] - freqs[:-1]
    assert (diffs <= 0).all(), "Frequencies should be non-increasing"


# ---------------------------------------------------------------------------
# 4. linear_position_interpolation — output == input / scale_factor
# ---------------------------------------------------------------------------
def test_linear_interpolation_values():
    head_dim = 32
    scale = 4.0
    base_freqs = compute_rope_freqs(head_dim)
    scaled = linear_position_interpolation(base_freqs, scale)
    expected = base_freqs / scale
    assert torch.allclose(scaled, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. ntk_aware_interpolation — shape
# ---------------------------------------------------------------------------
def test_ntk_aware_interpolation_shape():
    head_dim = 64
    freqs = ntk_aware_interpolation(head_dim, scale_factor=4.0, base_theta=10000.0)
    assert freqs.shape == (head_dim // 2,)


# ---------------------------------------------------------------------------
# 6. ntk_aware_interpolation — freqs < standard for scale > 1
# ---------------------------------------------------------------------------
def test_ntk_aware_interpolation_lower_than_standard():
    head_dim = 64
    base_theta = 10000.0
    scale = 4.0
    standard = compute_rope_freqs(head_dim, theta=base_theta)
    ntk = ntk_aware_interpolation(head_dim, scale_factor=scale, base_theta=base_theta)
    # Scaled theta is larger, so NTK freqs should be <= standard freqs.
    # Index 0 (i=0) is always 1.0 regardless of theta (theta^0 = 1), so use <=.
    assert (ntk <= standard).all(), "NTK freqs should be <= standard freqs"
    # All higher-index freqs should be strictly smaller
    assert (ntk[1:] < standard[1:]).all(), "NTK freqs[1:] should be strictly < standard freqs[1:]"


# ---------------------------------------------------------------------------
# 7. yarn_interpolation — shape
# ---------------------------------------------------------------------------
def test_yarn_interpolation_shape():
    head_dim = 64
    freqs = yarn_interpolation(
        head_dim, scale_factor=4.0, base_theta=10000.0, original_max_len=2048
    )
    assert freqs.shape == (head_dim // 2,)


# ---------------------------------------------------------------------------
# 8. yarn_interpolation — output between NTK and linear bounds
# ---------------------------------------------------------------------------
def test_yarn_interpolation_between_bounds():
    head_dim = 64
    scale = 4.0
    theta = 10000.0
    orig_len = 2048

    standard = compute_rope_freqs(head_dim, theta=theta)
    linear = linear_position_interpolation(standard, scale)
    ntk = ntk_aware_interpolation(head_dim, scale, theta)
    yarn = yarn_interpolation(head_dim, scale, theta, original_max_len=orig_len)

    lo = torch.minimum(linear, ntk)
    hi = torch.maximum(linear, ntk)

    assert (yarn >= lo - 1e-6).all(), "YaRN freqs should be >= min(linear, ntk)"
    assert (yarn <= hi + 1e-6).all(), "YaRN freqs should be <= max(linear, ntk)"


# ---------------------------------------------------------------------------
# 9. dynamic_ntk_interpolation — shape
# ---------------------------------------------------------------------------
def test_dynamic_ntk_interpolation_shape():
    head_dim = 64
    freqs = dynamic_ntk_interpolation(
        head_dim, seq_len=4096, original_max_len=2048, base_theta=10000.0
    )
    assert freqs.shape == (head_dim // 2,)


# ---------------------------------------------------------------------------
# 10. dynamic_ntk_interpolation — short seq → same as standard (no scaling)
# ---------------------------------------------------------------------------
def test_dynamic_ntk_no_scaling_within_limit():
    head_dim = 64
    base_theta = 10000.0
    original_max_len = 2048
    # seq_len <= original_max_len → scale_factor = 1 → same as standard RoPE
    dynamic = dynamic_ntk_interpolation(
        head_dim, seq_len=1024, original_max_len=original_max_len, base_theta=base_theta
    )
    standard = compute_rope_freqs(head_dim, theta=base_theta)
    assert torch.allclose(dynamic, standard, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. apply_rope_with_freqs — output same shape as input
# ---------------------------------------------------------------------------
def test_apply_rope_output_shape():
    B, H, T, D = 2, 4, 16, 64
    x = torch.randn(B, H, T, D)
    freqs = compute_rope_freqs(D)
    out = apply_rope_with_freqs(x, freqs)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 12. apply_rope_with_freqs — output != input (rotation is non-trivial)
# ---------------------------------------------------------------------------
def test_apply_rope_modifies_input():
    B, H, T, D = 2, 4, 16, 64
    x = torch.randn(B, H, T, D)
    freqs = compute_rope_freqs(D)
    out = apply_rope_with_freqs(x, freqs)
    # For T > 1 positions the rotation will change values
    assert not torch.allclose(out, x, atol=1e-4)


# ---------------------------------------------------------------------------
# 13. PositionInterpolator.get_freqs — correct shape
# ---------------------------------------------------------------------------
def test_position_interpolator_get_freqs_shape():
    head_dim = 64
    for method in ("linear", "ntk", "yarn", "dynamic"):
        cfg = PIConfig(method=method, scale_factor=4.0, base_theta=10000.0,
                       original_max_len=2048, extended_max_len=8192)
        interp = PositionInterpolator(cfg, head_dim)
        freqs = interp.get_freqs(seq_len=4096)
        assert freqs.shape == (head_dim // 2,), (
            f"Method {method!r} returned wrong shape: {freqs.shape}"
        )


# ---------------------------------------------------------------------------
# 14. PositionInterpolator.apply — output same shape as input
# ---------------------------------------------------------------------------
def test_position_interpolator_apply_shape():
    B, H, T, D = 2, 4, 32, 64
    x = torch.randn(B, H, T, D)
    cfg = PIConfig(method="yarn", scale_factor=4.0, base_theta=10000.0,
                   original_max_len=2048, extended_max_len=8192)
    interp = PositionInterpolator(cfg, head_dim=D)
    out = interp.apply(x)
    assert out.shape == x.shape
