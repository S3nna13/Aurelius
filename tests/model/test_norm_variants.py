"""Tests for src/model/norm_variants.py"""

import pytest
import torch
import torch.nn as nn

from src.model.norm_variants import (
    DynamicTanh,
    GroupNorm1D,
    QKNorm,
    ScaleNorm,
    replace_norms,
)
from src.model.rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# GroupNorm1D
# ---------------------------------------------------------------------------


def test_group_norm_1d_output_shape():
    layer = GroupNorm1D(num_channels=64, num_groups=8)
    x = torch.randn(2, 16, 64)
    out = layer(x)
    assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"


def test_group_norm_1d_requires_divisible():
    with pytest.raises(AssertionError):
        GroupNorm1D(num_channels=64, num_groups=7)


# ---------------------------------------------------------------------------
# DynamicTanh
# ---------------------------------------------------------------------------


def test_dynamic_tanh_output_shape():
    layer = DynamicTanh(d_model=64)
    # Test with a 3-D tensor (B, T, C)
    x = torch.randn(4, 10, 64)
    out = layer(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    # Also test with a 2-D tensor (T, C)
    x2 = torch.randn(10, 64)
    out2 = layer(x2)
    assert out2.shape == x2.shape


def test_dynamic_tanh_bounded():
    """Output magnitude is bounded by max(|gamma|) because tanh ∈ (-1, 1)."""
    layer = DynamicTanh(d_model=64)
    # Use a large-magnitude input to saturate tanh
    x = torch.randn(8, 32, 64) * 100
    out = layer(x)
    max_gamma = layer.gamma.abs().max().item()
    # Each element: |output| <= max_gamma * |tanh(α*x)| + |beta|
    # For saturated tanh, tanh → ±1, so |output| ≈ |gamma| + |beta|
    # We only test the gamma-dominated bound (beta starts at 0)
    assert out.abs().max().item() < max_gamma * 2, (
        "DyT output should be roughly bounded by gamma when tanh is saturated"
    )


def test_dynamic_tanh_learnable_alpha():
    layer = DynamicTanh(d_model=32)
    param_names = [name for name, _ in layer.named_parameters()]
    assert "alpha" in param_names, "alpha must be a learnable parameter"
    # Confirm it is a scalar (0-dim or 1-element)
    assert layer.alpha.numel() == 1, "alpha must be a scalar parameter"


# ---------------------------------------------------------------------------
# QKNorm
# ---------------------------------------------------------------------------


def test_qk_norm_output_shapes():
    """q (2,4,8,16) and k (2,2,8,16) — n_kv_heads < n_heads (GQA)."""
    n_heads, n_kv_heads, T, head_dim = 4, 2, 8, 16
    layer = QKNorm(n_heads=n_heads, head_dim=head_dim)
    q = torch.randn(2, n_heads, T, head_dim)
    k = torch.randn(2, n_kv_heads, T, head_dim)
    q_out, k_out = layer(q, k)
    assert q_out.shape == q.shape, f"q shape mismatch: {q_out.shape}"
    assert k_out.shape == k.shape, f"k shape mismatch: {k_out.shape}"


def test_qk_norm_unit_norm_before_scale():
    """After zeroing out the learned scale, Q should have unit RMS norm."""
    n_heads, T, head_dim = 4, 8, 16
    layer = QKNorm(n_heads=n_heads, head_dim=head_dim)

    # Zero the learned scales so output == normalize(q)
    with torch.no_grad():
        layer.scale_q.fill_(1.0)
        layer.scale_k.fill_(1.0)

    q = torch.randn(2, n_heads, T, head_dim)
    k = torch.randn(2, n_heads, T, head_dim)
    q_out, _ = layer(q, k)

    # RMS norm along head_dim should be ~1 for every (b, h, t) position
    rms = q_out.pow(2).mean(dim=-1).sqrt()  # (B, n_heads, T)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
        f"Expected unit RMS norm, max deviation: {(rms - 1).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# ScaleNorm
# ---------------------------------------------------------------------------


def test_scale_norm_output_shape():
    layer = ScaleNorm(d_model=64)
    x = torch.randn(2, 8, 64)
    out = layer(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"


def test_scale_norm_single_parameter():
    layer = ScaleNorm(d_model=64)
    params = list(layer.parameters())
    assert len(params) == 1, f"ScaleNorm should have exactly 1 parameter, got {len(params)}"
    assert params[0].numel() == 1, "The single parameter g must be a scalar"


# ---------------------------------------------------------------------------
# replace_norms
# ---------------------------------------------------------------------------


def _build_tiny_model(d_model: int = 32, n_norms: int = 3) -> nn.Module:
    """Build a tiny sequential model with n_norms RMSNorm layers."""
    layers: list[nn.Module] = []
    for _ in range(n_norms):
        layers.append(nn.Linear(d_model, d_model))
        layers.append(RMSNorm(d_model))
    return nn.Sequential(*layers)


def test_replace_norms_count():
    model = _build_tiny_model(d_model=32, n_norms=3)

    # Confirm we start with exactly 3 RMSNorm instances
    initial = sum(1 for _, m in model.named_modules() if isinstance(m, RMSNorm))
    assert initial == 3

    count = replace_norms(model, norm_type="dyt")
    assert count == 3, f"Expected 3 replacements, got {count}"

    # Confirm no RMSNorm instances remain
    remaining = sum(1 for _, m in model.named_modules() if isinstance(m, RMSNorm))
    assert remaining == 0, f"Expected 0 RMSNorm after replacement, got {remaining}"

    # Confirm DynamicTanh modules are present
    dyt_count = sum(1 for _, m in model.named_modules() if isinstance(m, DynamicTanh))
    assert dyt_count == 3, f"Expected 3 DynamicTanh after replacement, got {dyt_count}"
