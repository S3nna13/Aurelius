"""Unit tests for InfiniAttention (Munkhdalai 2024)."""

from __future__ import annotations

import time

import pytest
import torch

from src.longcontext.infini_attention import InfiniAttention, _elu_plus_one


# Tiny config shared across tests.
B, H, S, D = 2, 2, 8, 8
D_MODEL = H * D


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    return q, k, v


def test_forward_output_shape():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs()
    out = layer(q, k, v)
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()


def test_memory_updates_across_calls_without_reset():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(1)
    layer(q, k, v)
    M1 = layer.memory_M.clone()
    z1 = layer.memory_z.clone()
    layer(q, k, v)
    # Memory must have accumulated — not equal after a second identical call.
    assert not torch.allclose(M1, layer.memory_M)
    assert not torch.allclose(z1, layer.memory_z)


def test_reset_memory_zeroes_state():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(2)
    layer(q, k, v)
    assert layer.memory_M.abs().sum() > 0
    layer.reset_memory()
    assert torch.all(layer.memory_M == 0)
    assert torch.all(layer.memory_z == 0)


def test_detach_memory_breaks_autograd_graph():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(3)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    layer(q, k, v)
    assert layer.memory_M.grad_fn is not None
    layer.detach_memory()
    assert layer.memory_M.grad_fn is None
    assert layer.memory_z.grad_fn is None
    assert layer.memory_M.requires_grad is False


def test_gradient_flow_through_beta():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(4)
    out = layer(q, k, v)
    out.sum().backward()
    assert layer.beta.grad is not None
    assert torch.isfinite(layer.beta.grad).all()
    # With nonzero memory contribution and random inputs, grad shouldn't be all-zero.
    assert layer.beta.grad.abs().sum() > 0


def test_determinism_with_seed():
    layer_a = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    layer_b = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(5)
    out_a = layer_a(q, k, v)
    out_b = layer_b(q, k, v)
    assert torch.allclose(out_a, out_b)


def test_no_nan_inf_over_three_sequential_segments():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    for seed in range(3):
        q, k, v = _make_inputs(10 + seed)
        out = layer(q, k, v)
        assert torch.isfinite(out).all()
        assert torch.isfinite(layer.memory_M).all()
        assert torch.isfinite(layer.memory_z).all()


def test_reset_memory_kwarg_in_forward():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    # Two distinct segments, so that without reset the second output depends
    # on the first; with reset, the second output is segment-2-only.
    q1, k1, v1 = _make_inputs(6)
    q2, k2, v2 = _make_inputs(7)
    # With memory: seed seg1 then run seg2 on top.
    layer(q1, k1, v1)
    out_accum = layer(q2, k2, v2)
    # Fresh run of seg2 alone.
    fresh = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    out_fresh = fresh(q2, k2, v2)
    assert not torch.allclose(out_accum, out_fresh, atol=1e-4)
    # Now test reset_memory=True: prime with seg1, then reset+run seg2.
    layer2 = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    layer2(q1, k1, v1)
    out_reset = layer2(q2, k2, v2, reset_memory=True)
    assert torch.allclose(out_reset, out_fresh, atol=1e-6)


def test_causal_false_works():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q, k, v = _make_inputs(7)
    out = layer(q, k, v, causal=False)
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()


def test_elu_plus_one_is_positive_elementwise():
    x = torch.randn(64) * 5.0  # wide range
    y = _elu_plus_one(x)
    assert (y > 0).all()
    # Also check extreme-negative: elu asymptotes at -1, so elu+1 -> 0+.
    very_neg = torch.full((16,), -50.0)
    assert (_elu_plus_one(very_neg) >= 0).all()


def test_long_sequence_three_segments_under_one_second():
    S_long = 64
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    t0 = time.perf_counter()
    for seed in range(3):
        torch.manual_seed(seed)
        q = torch.randn(B, H, S_long, D)
        k = torch.randn(B, H, S_long, D)
        v = torch.randn(B, H, S_long, D)
        out = layer(q, k, v)
        assert torch.isfinite(out).all()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"three segments of 64 took {elapsed:.3f}s"


def test_shape_mismatch_raises():
    layer = InfiniAttention(d_model=D_MODEL, n_heads=H, head_dim=D)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v_bad = torch.randn(B, H, S, D + 1)
    with pytest.raises(ValueError):
        layer(q, k, v_bad)
    # Wrong head count.
    q_wrongH = torch.randn(B, H + 1, S, D)
    k_wrongH = torch.randn(B, H + 1, S, D)
    v_wrongH = torch.randn(B, H + 1, S, D)
    with pytest.raises(ValueError):
        layer(q_wrongH, k_wrongH, v_wrongH)
    # Non-tensor.
    with pytest.raises(TypeError):
        layer("not a tensor", k, torch.randn(B, H, S, D))
    # Wrong rank.
    with pytest.raises(ValueError):
        layer(torch.randn(B, H, S), k, torch.randn(B, H, S, D))


def test_constructor_validates_dims():
    with pytest.raises(ValueError):
        InfiniAttention(d_model=16, n_heads=0, head_dim=8)
    with pytest.raises(ValueError):
        InfiniAttention(d_model=17, n_heads=2, head_dim=8)  # 2*8 != 17
    with pytest.raises(ValueError):
        InfiniAttention(d_model=-1, n_heads=2, head_dim=8)
