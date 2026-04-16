"""Tests for src/model/activations_norms.py.

Uses tiny dimensions throughout so the suite runs quickly on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.activations_norms import (
    swish,
    gelu_tanh,
    SwiGLU,
    GeGLU,
    RMSNorm,
    ScaleNorm,
    CRMSNorm,
    compute_activation_stats,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
B = 2
T = 4
D_MODEL = 16
D_FFN = 32


# ---------------------------------------------------------------------------
# swish
# ---------------------------------------------------------------------------

def test_swish_output_shape():
    """swish output shape must match input shape."""
    x = torch.randn(B, T, D_MODEL)
    out = swish(x)
    assert out.shape == x.shape


def test_swish_at_zero():
    """swish(0) == 0 because 0 * sigmoid(0) == 0."""
    x = torch.zeros(3, 3)
    out = swish(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_swish_positive_for_large_positive_input():
    """swish should be positive for large positive inputs."""
    x = torch.tensor([5.0, 10.0, 100.0])
    out = swish(x)
    assert (out > 0).all()


def test_swish_matches_silu():
    """swish should be numerically identical to F.silu."""
    import torch.nn.functional as F
    x = torch.randn(8, 8)
    assert torch.allclose(swish(x), F.silu(x), atol=1e-6)


# ---------------------------------------------------------------------------
# gelu_tanh
# ---------------------------------------------------------------------------

def test_gelu_tanh_output_shape():
    """gelu_tanh output shape must match input shape."""
    x = torch.randn(B, T, D_MODEL)
    out = gelu_tanh(x)
    assert out.shape == x.shape


def test_gelu_tanh_at_zero():
    """gelu_tanh(0) should be approximately 0."""
    x = torch.zeros(4)
    out = gelu_tanh(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_gelu_tanh_output_finite():
    """gelu_tanh outputs should be finite for normal inputs."""
    x = torch.randn(16, 16)
    out = gelu_tanh(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

def test_swiglu_output_shape():
    """SwiGLU forward must return (B, T, D_MODEL)."""
    model = SwiGLU(D_MODEL, D_FFN)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


def test_swiglu_output_finite():
    """SwiGLU output values must all be finite."""
    model = SwiGLU(D_MODEL, D_FFN)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all()


def test_swiglu_no_bias():
    """SwiGLU linear projections should have no bias parameters."""
    model = SwiGLU(D_MODEL, D_FFN)
    assert model.W.bias is None
    assert model.V.bias is None
    assert model.W2.bias is None


def test_swiglu_batch_independence():
    """Each batch element should be processed independently."""
    model = SwiGLU(D_MODEL, D_FFN)
    model.eval()
    x = torch.randn(B, T, D_MODEL)
    out_full = model(x)
    out_single = model(x[:1])
    assert torch.allclose(out_full[:1], out_single, atol=1e-5)


# ---------------------------------------------------------------------------
# GeGLU
# ---------------------------------------------------------------------------

def test_geglu_output_shape():
    """GeGLU forward must return (B, T, D_MODEL)."""
    model = GeGLU(D_MODEL, D_FFN)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


def test_geglu_output_finite():
    """GeGLU output values must all be finite."""
    model = GeGLU(D_MODEL, D_FFN)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def test_rmsnorm_output_shape():
    """RMSNorm output shape must match input shape."""
    norm = RMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_rms():
    """RMSNorm output (with default gamma=1) should have RMS close to 1."""
    norm = RMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_gamma_is_learnable_parameter():
    """RMSNorm gamma must be an nn.Parameter of shape (d_model,)."""
    norm = RMSNorm(D_MODEL)
    assert isinstance(norm.gamma, nn.Parameter)
    assert norm.gamma.shape == (D_MODEL,)


def test_rmsnorm_gamma_scales_output():
    """Doubling gamma should double the output."""
    norm = RMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        out1 = norm(x)
        norm.gamma.fill_(2.0)
        out2 = norm(x)
    assert torch.allclose(out2, 2.0 * out1, atol=1e-5)


# ---------------------------------------------------------------------------
# ScaleNorm
# ---------------------------------------------------------------------------

def test_scalenorm_output_shape():
    """ScaleNorm output shape must match input shape."""
    norm = ScaleNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = norm(x)
    assert out.shape == x.shape


def test_scalenorm_output_finite():
    """ScaleNorm output values must be finite."""
    norm = ScaleNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = norm(x)
    assert torch.isfinite(out).all()


def test_scalenorm_g_initialized_to_sqrt_d_model():
    """ScaleNorm scalar g must be initialized to sqrt(d_model)."""
    norm = ScaleNorm(D_MODEL)
    expected = math.sqrt(D_MODEL)
    assert abs(norm.g.item() - expected) < 1e-5


def test_scalenorm_l2_norm_matches_g():
    """Output L2 norm of each vector should equal g (approximately)."""
    norm = ScaleNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL) + 1.0
    out = norm(x)
    l2_norms = out.norm(dim=-1)
    g_val = norm.g.item()
    assert torch.allclose(l2_norms, torch.full_like(l2_norms, g_val), atol=1e-3)


# ---------------------------------------------------------------------------
# CRMSNorm
# ---------------------------------------------------------------------------

def test_crmsnorm_output_shape():
    """CRMSNorm output shape must be (B, T, D_MODEL)."""
    norm = CRMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    scale = torch.zeros(B, D_MODEL)
    shift = torch.zeros(B, D_MODEL)
    out = norm(x, scale, shift)
    assert out.shape == (B, T, D_MODEL)


def test_crmsnorm_zero_scale_shift_equals_rmsnorm():
    """With scale=0 and shift=0, CRMSNorm should equal plain RMSNorm (gamma=1)."""
    crmsnorm = CRMSNorm(D_MODEL, eps=1e-6)
    rmsnorm = RMSNorm(D_MODEL, eps=1e-6)
    with torch.no_grad():
        rmsnorm.gamma.fill_(1.0)
    x = torch.randn(B, T, D_MODEL)
    scale = torch.zeros(B, D_MODEL)
    shift = torch.zeros(B, D_MODEL)
    out_c = crmsnorm(x, scale, shift)
    out_r = rmsnorm(x)
    assert torch.allclose(out_c, out_r, atol=1e-5)


def test_crmsnorm_no_learnable_params():
    """CRMSNorm should have no learnable parameters."""
    norm = CRMSNorm(D_MODEL)
    params = list(norm.parameters())
    assert len(params) == 0


def test_crmsnorm_broadcast_1d_scale_shift():
    """CRMSNorm should accept 1-D scale/shift of shape (d_model,)."""
    norm = CRMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    scale = torch.zeros(D_MODEL)
    shift = torch.zeros(D_MODEL)
    out = norm(x, scale, shift)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# compute_activation_stats
# ---------------------------------------------------------------------------

def test_compute_activation_stats_keys():
    """compute_activation_stats must return all required keys."""
    x = torch.randn(4, 8)
    stats = compute_activation_stats(x)
    required_keys = {"mean", "std", "fraction_positive", "max_abs"}
    assert required_keys == set(stats.keys())


def test_compute_activation_stats_fraction_positive_range():
    """fraction_positive must be in [0, 1]."""
    x = torch.randn(100, 100)
    stats = compute_activation_stats(x)
    assert 0.0 <= stats["fraction_positive"] <= 1.0


def test_compute_activation_stats_all_positive():
    """All-positive tensor should have fraction_positive == 1.0."""
    x = torch.ones(10, 10)
    stats = compute_activation_stats(x)
    assert stats["fraction_positive"] == pytest.approx(1.0)


def test_compute_activation_stats_max_abs():
    """max_abs should match manual calculation."""
    x = torch.tensor([-5.0, 1.0, 3.0])
    stats = compute_activation_stats(x)
    assert stats["max_abs"] == pytest.approx(5.0, abs=1e-5)


def test_compute_activation_stats_values_are_floats():
    """All returned stats values must be Python floats."""
    x = torch.randn(8, 8)
    stats = compute_activation_stats(x)
    for key, val in stats.items():
        assert isinstance(val, float), f"{key} is not a float: {type(val)}"
