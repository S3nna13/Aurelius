"""Unit tests for src.longcontext.kv_compression."""

from __future__ import annotations

import math

import pytest
import torch

from src.longcontext import (
    LONGCONTEXT_STRATEGY_REGISTRY,
    CompressedKV,
    KVInt8Compressor,
    quantize_per_head_symmetric,
)


B, H, S, D = 2, 4, 8, 16


def _make_kv(seed: int = 0, s: int = S, b: int = B, h: int = H, d: int = D):
    torch.manual_seed(seed)
    k = torch.randn(b, h, s, d, dtype=torch.float32)
    v = torch.randn(b, h, s, d, dtype=torch.float32)
    return k, v


def test_registry_registration():
    assert "kv_int8" in LONGCONTEXT_STRATEGY_REGISTRY
    assert LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"] is KVInt8Compressor


def test_round_trip_shapes_and_tolerance():
    k, v = _make_kv()
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    packed = c.compress(k, v)
    kd, vd = c.decompress(packed)
    assert kd.shape == k.shape
    assert vd.shape == v.shape
    assert torch.allclose(kd, k, atol=0.05)
    assert torch.allclose(vd, v, atol=0.05)
    # Relative error sanity: absmax of err bounded by scale
    err_k = (kd - k).abs().max().item()
    assert err_k < 0.05


def test_dtypes_int8_in_and_float_out():
    k, v = _make_kv()
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    packed = c.compress(k, v)
    assert packed.k_q.dtype == torch.int8
    assert packed.v_q.dtype == torch.int8
    assert packed.k_scale.dtype == torch.float32
    assert packed.v_scale.dtype == torch.float32
    kd, vd = c.decompress(packed)
    assert kd.dtype == k.dtype
    assert vd.dtype == v.dtype


def test_scales_positive_and_finite():
    k, v = _make_kv()
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    packed = c.compress(k, v)
    assert torch.all(packed.k_scale > 0)
    assert torch.all(packed.v_scale > 0)
    assert torch.isfinite(packed.k_scale).all()
    assert torch.isfinite(packed.v_scale).all()
    # Scale shape: [B, H]
    assert tuple(packed.k_scale.shape) == (B, H)
    assert tuple(packed.v_scale.shape) == (B, H)


def test_determinism_under_seed():
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    k1, v1 = _make_kv(seed=0)
    k2, v2 = _make_kv(seed=0)
    p1 = c.compress(k1, v1)
    p2 = c.compress(k2, v2)
    assert torch.equal(p1.k_q, p2.k_q)
    assert torch.equal(p1.v_q, p2.v_q)
    assert torch.equal(p1.k_scale, p2.k_scale)


def test_append_matches_fresh_compress():
    # compress(K[:, :, :S1, :]) + append(K[:, :, S1:, :]) ~= compress(K[:, :, :S, :])
    k, v = _make_kv(seed=1, s=S)
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    s1 = 5
    p_full = c.compress(k, v)
    p_part = c.compress(k[:, :, :s1, :], v[:, :, :s1, :])
    p_stream = c.append(p_part, k[:, :, s1:, :], v[:, :, s1:, :])

    assert p_stream.k_q.shape == p_full.k_q.shape
    assert p_stream.v_q.shape == p_full.v_q.shape

    k_full, v_full = c.decompress(p_full)
    k_stream, v_stream = c.decompress(p_stream)
    # Streaming uses a joint scale that equals absmax over full seq
    # (old scale OR new absmax, whichever is larger), so accuracy should
    # be comparable to a single-shot compress.
    assert torch.allclose(k_stream, k, atol=0.05)
    assert torch.allclose(v_stream, v, atol=0.05)
    # And close to the one-shot decompress (streaming rescales history
    # codes by integer rounding, so allow ~one LSB of extra slack).
    assert torch.allclose(k_stream, k_full, atol=0.05)
    assert torch.allclose(v_stream, v_full, atol=0.05)


def test_edge_case_single_token():
    k, v = _make_kv(seed=2, s=1)
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    assert p.seq_len == 1
    kd, vd = c.decompress(p)
    assert torch.allclose(kd, k, atol=0.05)


def test_edge_case_batch_one():
    k, v = _make_kv(seed=3, b=1)
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    kd, vd = c.decompress(p)
    assert kd.shape == (1, H, S, D)
    assert torch.allclose(kd, k, atol=0.05)


def test_edge_case_all_zero_no_nan():
    k = torch.zeros(B, H, S, D, dtype=torch.float32)
    v = torch.zeros(B, H, S, D, dtype=torch.float32)
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    assert torch.isfinite(p.k_scale).all()
    assert torch.all(p.k_scale > 0)
    kd, vd = c.decompress(p)
    assert not torch.isnan(kd).any()
    assert not torch.isinf(kd).any()
    assert torch.equal(kd, torch.zeros_like(kd))
    assert torch.equal(vd, torch.zeros_like(vd))


def test_edge_case_extreme_values():
    k = torch.full((B, H, S, D), 1e4, dtype=torch.float32)
    k[0, 0, 0, 0] = -1e4
    v = -k.clone()
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    kd, vd = c.decompress(p)
    assert torch.isfinite(kd).all()
    assert torch.isfinite(vd).all()
    # Relative error should be < ~1% for uniform-extreme inputs.
    rel = ((kd - k).abs() / k.abs().clamp_min(1.0)).max().item()
    assert rel < 0.02


def test_no_nan_inf_in_decompress_random():
    for seed in range(5):
        k, v = _make_kv(seed=seed)
        c = KVInt8Compressor(head_dim=D, n_heads=H)
        p = c.compress(k, v)
        kd, vd = c.decompress(p)
        assert torch.isfinite(kd).all()
        assert torch.isfinite(vd).all()


def test_decompressed_requires_grad_false_documented():
    # Quantization is not differentiable; decompressed output must be a
    # detached inference artifact. Even if the *input* tracks grads, the
    # compressed buffer is int8 and the output must not require grad.
    k = torch.randn(B, H, S, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.float32, requires_grad=True)
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    kd, vd = c.decompress(p)
    assert kd.requires_grad is False
    assert vd.requires_grad is False


def test_long_sequence_no_oom():
    # Tiny D keeps this small but exercises the S=8192 path.
    long_s = 8192
    k = torch.randn(1, 2, long_s, 4, dtype=torch.float32)
    v = torch.randn(1, 2, long_s, 4, dtype=torch.float32)
    c = KVInt8Compressor(head_dim=4, n_heads=2)
    p = c.compress(k, v)
    assert p.k_q.shape == (1, 2, long_s, 4)
    assert p.k_q.dtype == torch.int8
    kd, _ = c.decompress(p)
    assert kd.shape == k.shape
    assert torch.isfinite(kd).all()


def test_helper_quantize_per_head_symmetric_shape():
    x = torch.randn(B, H, S, D)
    q, s = quantize_per_head_symmetric(x, dim=(-2, -1))
    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert tuple(s.shape) == (B, H)
    assert torch.all(s > 0)


def test_helper_rejects_non_float():
    with pytest.raises(TypeError):
        quantize_per_head_symmetric(torch.zeros(2, 2, dtype=torch.int32))


def test_shape_mismatch_raises():
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S + 1, D)
    with pytest.raises(ValueError):
        c.compress(k, v)


def test_fp8_stub_raises():
    from src.longcontext.kv_compression import KVFP8Compressor

    with pytest.raises(NotImplementedError):
        KVFP8Compressor(head_dim=D, n_heads=H)


def test_compressed_kv_nbytes_smaller_than_fp32():
    k, v = _make_kv()
    c = KVInt8Compressor(head_dim=D, n_heads=H)
    p = c.compress(k, v)
    fp32_bytes = k.numel() * 4 + v.numel() * 4
    assert p.nbytes() < fp32_bytes
    # Dominated by int8 buffers -> ~1/4 of fp32 plus tiny scales.
    assert p.nbytes() < fp32_bytes / 3
