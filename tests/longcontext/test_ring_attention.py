"""Unit tests for ring_attention."""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn.functional as F

from src.longcontext.ring_attention import RingAttention, ring_attention


B, H, S, D = 2, 4, 32, 8


def _reference_attention(q, k, v, causal=False, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        S_q, S_k = q.shape[-2], k.shape[-2]
        offset = S_k - S_q
        q_pos = torch.arange(S_q, device=q.device).unsqueeze(-1) + offset
        k_pos = torch.arange(S_k, device=q.device).unsqueeze(0)
        logits = logits.masked_fill(k_pos > q_pos, torch.finfo(q.dtype).min)
    if mask is not None:
        logits = logits + mask
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, v)


def _qkv(seed=0, s=S, d=D, dtype=torch.float32, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, s, d, generator=g, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, s, d, generator=g, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, H, s, d, generator=g, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def test_output_shape():
    q, k, v = _qkv()
    out = ring_attention(q, k, v, chunk_size=8)
    assert out.shape == (B, H, S, D)


def test_output_dtype_preserved():
    q, k, v = _qkv(dtype=torch.float64)
    out = ring_attention(q, k, v, chunk_size=8)
    assert out.dtype == torch.float64


def test_numerical_equivalence_to_reference():
    q, k, v = _qkv(dtype=torch.float64)
    ref = _reference_attention(q, k, v)
    got = ring_attention(q, k, v, chunk_size=8)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("chunk", [4, 8, 16, S])
def test_equivalence_across_chunk_sizes(chunk):
    q, k, v = _qkv(dtype=torch.float64)
    ref = _reference_attention(q, k, v)
    got = ring_attention(q, k, v, chunk_size=chunk)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_causal_matches_sdpa():
    q, k, v = _qkv(dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = ring_attention(q, k, v, chunk_size=8, causal=True)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_gradient_flow():
    q, k, v = _qkv(dtype=torch.float32, requires_grad=True)
    out = ring_attention(q, k, v, chunk_size=8)
    loss = out.square().mean()
    loss.backward()
    for t, name in [(q, "q"), (k, "k"), (v, "v")]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        assert t.grad.abs().sum() > 0, f"{name}.grad is all zero"


def test_determinism_with_manual_seed():
    torch.manual_seed(1234)
    q1, k1, v1 = _qkv(seed=7)
    o1 = ring_attention(q1, k1, v1, chunk_size=8)
    torch.manual_seed(1234)
    q2, k2, v2 = _qkv(seed=7)
    o2 = ring_attention(q2, k2, v2, chunk_size=8)
    torch.testing.assert_close(o1, o2, atol=0, rtol=0)


def test_edge_case_s_equals_1():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, 1, D, generator=g, dtype=torch.float64)
    k = torch.randn(B, H, 1, D, generator=g, dtype=torch.float64)
    v = torch.randn(B, H, 1, D, generator=g, dtype=torch.float64)
    ref = _reference_attention(q, k, v)
    got = ring_attention(q, k, v, chunk_size=4)
    assert got.shape == (B, H, 1, D)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


def test_edge_case_chunk_size_1():
    q, k, v = _qkv(dtype=torch.float64)
    ref = _reference_attention(q, k, v)
    got = ring_attention(q, k, v, chunk_size=1)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_edge_case_chunk_size_larger_than_s():
    q, k, v = _qkv(dtype=torch.float64)
    ref = _reference_attention(q, k, v)
    got = ring_attention(q, k, v, chunk_size=S * 4)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


def test_explicit_additive_mask():
    q, k, v = _qkv(dtype=torch.float64)
    mask = torch.zeros(B, H, S, S, dtype=torch.float64)
    # Forbid the last 5 keys entirely.
    mask[..., -5:] = float("-inf")
    ref = _reference_attention(q, k, v, mask=mask)
    got = ring_attention(q, k, v, chunk_size=8, mask=mask)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_no_nan_inf_with_extreme_inputs():
    q, k, v = _qkv(dtype=torch.float32)
    q = q * 1e4
    k = k * 1e4
    out = ring_attention(q, k, v, chunk_size=8)
    assert torch.isfinite(out).all(), "ring_attention produced non-finite output on large inputs"


def test_extrapolated_long_sequence_speed():
    torch.manual_seed(0)
    B2, H2, S2, D2 = 1, 2, 1024, 16
    q = torch.randn(B2, H2, S2, D2)
    k = torch.randn(B2, H2, S2, D2)
    v = torch.randn(B2, H2, S2, D2)
    # Warmup
    ring_attention(q, k, v, chunk_size=64)
    t0 = time.perf_counter()
    out = ring_attention(q, k, v, chunk_size=64)
    elapsed = time.perf_counter() - t0
    assert out.shape == (B2, H2, S2, D2)
    assert elapsed < 2.0, f"ring_attention S=1024 took {elapsed:.3f}s (>2s)"


def test_wrong_shape_raises():
    with pytest.raises((ValueError, TypeError)):
        # 3-D input
        ring_attention(torch.randn(H, S, D), torch.randn(H, S, D), torch.randn(H, S, D))
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D + 1)  # mismatched head dim
    v = torch.randn(B, H, S, D)
    with pytest.raises(ValueError):
        ring_attention(q, k, v)
    k2 = torch.randn(B, H, S, D)
    v2 = torch.randn(B, H, S + 1, D)  # mismatched k/v seq len
    with pytest.raises(ValueError):
        ring_attention(q, k2, v2)
    # Bad chunk size
    with pytest.raises(ValueError):
        ring_attention(q, k2, v2[:, :, :S, :], chunk_size=0)


def test_wrapper_class_matches_functional():
    q, k, v = _qkv(dtype=torch.float64)
    ra = RingAttention(chunk_size=8, causal=False)
    out_cls = ra(q, k, v)
    out_fn = ring_attention(q, k, v, chunk_size=8, causal=False)
    torch.testing.assert_close(out_cls, out_fn, atol=0, rtol=0)


def test_wrapper_class_causal():
    q, k, v = _qkv(dtype=torch.float32)
    ra = RingAttention(chunk_size=4, causal=True)
    out = ra(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
