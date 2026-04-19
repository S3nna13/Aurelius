"""Unit tests for Compressive Transformer memory."""

from __future__ import annotations

import time

import pytest
import torch

from src.longcontext.compressive_transformer import (
    CompressiveMemory,
    CompressiveMemoryState,
)


B, H, D = 2, 4, 16
RECENT = 8
COMP = 8
RATE = 2


def _mem(fn: str = "mean_pool", **kw) -> CompressiveMemory:
    m = CompressiveMemory(
        n_heads=H,
        head_dim=D,
        recent_size=RECENT,
        compressed_size=COMP,
        compression_rate=RATE,
        compression_fn=fn,
        **kw,
    )
    m.reset(batch_size=B, dtype=torch.float32)
    return m


def _kv(seq: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(B, H, seq, D, generator=g)
    v = torch.randn(B, H, seq, D, generator=g)
    return k, v


def test_reset_initializes_empty_state():
    m = _mem()
    s = m.state()
    assert s.recent_k is None and s.recent_v is None
    assert s.compressed_k is None and s.compressed_v is None


def test_update_within_recent_grows_recent():
    m = _mem()
    k, v = _kv(4)
    m.update(k, v)
    s = m.state()
    assert s.recent_k.shape == (B, H, 4, D)
    assert s.compressed_k is None
    # Grow further but stay within recent_size.
    k2, v2 = _kv(4, seed=1)
    m.update(k2, v2)
    s = m.state()
    assert s.recent_k.shape == (B, H, 8, D)
    assert s.compressed_k is None


def test_update_overflow_evicts_into_compressed():
    m = _mem()
    # Push 10 tokens; recent_size=8 -> evict 2 tokens, compression_rate=2 -> 1 compressed token.
    k, v = _kv(10)
    m.update(k, v)
    s = m.state()
    assert s.recent_k.shape == (B, H, 8, D)
    assert s.compressed_k.shape == (B, H, 1, D)
    assert s.compressed_v.shape == (B, H, 1, D)


def test_compressed_shape_matches_expected():
    m = _mem()
    # 12 tokens -> evict 4 -> 4//2 = 2 compressed.
    k, v = _kv(12)
    m.update(k, v)
    s = m.state()
    assert s.compressed_k.shape == (B, H, 2, D)
    assert s.recent_k.shape == (B, H, 8, D)


def test_mean_pool_produces_average():
    m = _mem("mean_pool")
    k, v = _kv(10)  # evict first 2 -> 1 compressed token = mean of first 2.
    expected_k = k[:, :, :2, :].mean(dim=2, keepdim=True)
    expected_v = v[:, :, :2, :].mean(dim=2, keepdim=True)
    m.update(k, v)
    s = m.state()
    assert torch.allclose(s.compressed_k, expected_k, atol=1e-6)
    assert torch.allclose(s.compressed_v, expected_v, atol=1e-6)


def test_max_pool_produces_max():
    m = _mem("max_pool")
    k, v = _kv(10)
    expected_k = k[:, :, :2, :].amax(dim=2, keepdim=True)
    expected_v = v[:, :, :2, :].amax(dim=2, keepdim=True)
    m.update(k, v)
    s = m.state()
    assert torch.allclose(s.compressed_k, expected_k, atol=1e-6)
    assert torch.allclose(s.compressed_v, expected_v, atol=1e-6)


def test_conv1d_is_trainable():
    m = _mem("conv1d")
    params = list(m.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
    # Train a step: compress through update and backprop.
    k, v = _kv(10)
    m.update(k, v)
    s = m.state()
    loss = s.compressed_k.pow(2).sum() + s.compressed_v.pow(2).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def test_concatenated_kv_shape():
    m = _mem()
    k, v = _kv(12)
    m.update(k, v)
    ck, cv = m.concatenated_kv()
    # 2 compressed + 8 recent = 10
    assert ck.shape == (B, H, 10, D)
    assert cv.shape == (B, H, 10, D)


def test_determinism():
    torch.manual_seed(123)
    m1 = _mem()
    k, v = _kv(14, seed=7)
    m1.update(k, v)
    ck1, cv1 = m1.concatenated_kv()

    torch.manual_seed(123)
    m2 = _mem()
    k2, v2 = _kv(14, seed=7)
    m2.update(k2, v2)
    ck2, cv2 = m2.concatenated_kv()

    assert torch.equal(ck1, ck2)
    assert torch.equal(cv1, cv2)


def test_no_nan_inf():
    m = _mem("mean_pool")
    k, v = _kv(20)
    m.update(k, v)
    ck, cv = m.concatenated_kv()
    assert torch.isfinite(ck).all()
    assert torch.isfinite(cv).all()


def test_overflow_beyond_compressed_evicts_oldest():
    # recent=8, comp=8, rate=2. Push one big block so we get > COMP compressed tokens.
    m = _mem()
    # Push 8 (fills recent) then push 20 more -> evicts 20 tokens -> 10 compressed.
    # But cap is 8, so oldest 2 compressed tokens get dropped.
    k0, v0 = _kv(8, seed=0)
    m.update(k0, v0)
    k1, v1 = _kv(20, seed=1)
    # After update: recent=[k0|k1]=28, overflow=20 -> evict first 20 = [k0(8)|k1[:12]].
    evicted_k = torch.cat([k0, k1[:, :, :12, :]], dim=2)
    expected_all = evicted_k.reshape(B, H, 10, 2, D).mean(dim=3)
    m.update(k1, v1)
    s = m.state()
    assert s.compressed_k.shape == (B, H, COMP, D)  # capped at 8
    # The retained compressed tokens should be the *last* 8 of the 10 computed.
    assert torch.allclose(s.compressed_k, expected_all[:, :, -COMP:, :], atol=1e-6)


def test_invalid_compression_fn_raises():
    with pytest.raises(ValueError):
        CompressiveMemory(
            n_heads=H, head_dim=D, recent_size=RECENT, compressed_size=COMP,
            compression_rate=RATE, compression_fn="bogus",
        )


def test_batch_mismatch_after_reset_raises():
    m = _mem()
    k = torch.randn(B + 1, H, 4, D)
    v = torch.randn(B + 1, H, 4, D)
    with pytest.raises(ValueError):
        m.update(k, v)


def test_eviction_indivisible_raises():
    # recent=8, rate=4 -> pushing 9 tokens evicts 1, not divisible by 4.
    m = CompressiveMemory(
        n_heads=H, head_dim=D, recent_size=8, compressed_size=8,
        compression_rate=4, compression_fn="mean_pool",
    )
    m.reset(batch_size=B)
    k = torch.randn(B, H, 9, D)
    v = torch.randn(B, H, 9, D)
    with pytest.raises(ValueError):
        m.update(k, v)


def test_2048_step_update_fast():
    m = _mem("mean_pool")
    t0 = time.perf_counter()
    for i in range(2048):
        k = torch.randn(B, H, 2, D)
        v = torch.randn(B, H, 2, D)
        m.update(k, v)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"2048-step update took {elapsed:.3f}s"
    s = m.state()
    assert s.recent_k.shape == (B, H, RECENT, D)
    assert s.compressed_k.shape == (B, H, COMP, D)
