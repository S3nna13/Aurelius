"""Unit tests for src/longcontext/kv_cache_quantization.py (KIVI INT4)."""

from __future__ import annotations

import time

import pytest
import torch

from src.longcontext.kv_cache_quantization import (
    KIVIQuantizer,
    pack_int4,
    unpack_int4,
)

# Tiny canonical config for these tests.
N_HEADS = 2
HEAD_DIM = 16
GROUP_SIZE = 16


def _rand_kv(B=2, H=N_HEADS, S=32, D=HEAD_DIM, seed=0, scale=0.5):
    # Scale is chosen so per-group span is small enough for INT4 bin-centre
    # reconstruction (range/30) to keep max error below 0.15, matching the
    # KIVI spec bound. Real post-LN activations are typically in this range.
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(B, H, S, D, generator=g, dtype=torch.float32) * scale
    v = torch.randn(B, H, S, D, generator=g, dtype=torch.float32) * scale
    return k, v


# ---------------------------------------------------------------------------
# 1. shape round-trip preserved
# ---------------------------------------------------------------------------


def test_shape_round_trip():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=2, S=64)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.shape == k.shape
    assert v2.shape == v.shape


# ---------------------------------------------------------------------------
# 2. dtype round-trip: decompressed is float
# ---------------------------------------------------------------------------


def test_dtype_round_trip_is_float():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv()
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.dtype.is_floating_point
    assert v2.dtype.is_floating_point


# ---------------------------------------------------------------------------
# 3. int4 range: packed uint8 buffer is half the size of int8 buffer
# ---------------------------------------------------------------------------


def test_int4_packed_buffer_is_half_of_int8():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=2, S=32)
    c = q.compress(k, v)
    # uint8 with two values per byte, vs hypothetical int8 with one per byte.
    n_elems = k.numel()
    assert c.k_q.dtype == torch.uint8
    assert c.v_q.dtype == torch.uint8
    # packed numel * 2 == original numel
    assert c.k_q.numel() * 2 == n_elems
    assert c.v_q.numel() * 2 == n_elems
    # packed byte count is exactly half of an equivalent int8 buffer's bytes
    int8_bytes = n_elems  # int8 element_size == 1
    assert c.k_q.numel() * c.k_q.element_size() == int8_bytes // 2
    assert c.v_q.numel() * c.v_q.element_size() == int8_bytes // 2


# ---------------------------------------------------------------------------
# 4. compression error atol<=0.15 on random tensors
# ---------------------------------------------------------------------------


def test_compression_error_bound():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=2, S=64, seed=7)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    # INT4 is lossier than INT8 but per-group asymmetric keeps max err bounded.
    assert torch.allclose(k, k2, atol=0.15)
    assert torch.allclose(v, v2, atol=0.15)


# ---------------------------------------------------------------------------
# 5. edge cases: B=1, H=1, S=1, D=group_size
# ---------------------------------------------------------------------------


def test_edge_case_batch_one():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=1, S=32)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.shape == k.shape
    assert torch.allclose(k, k2, atol=0.15)


def test_edge_case_single_head():
    # group_size must still divide head_dim and be even
    q = KIVIQuantizer(n_heads=1, head_dim=HEAD_DIM, group_size=GROUP_SIZE)
    k, v = _rand_kv(B=2, H=1, S=32)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.shape == k.shape
    assert torch.allclose(v, v2, atol=0.15)


def test_edge_case_minimum_sequence():
    # Minimum S that can support per-token grouping is group_size itself.
    # Use group_size=2 so we can exercise the smallest valid S.
    q = KIVIQuantizer(n_heads=1, head_dim=HEAD_DIM, group_size=2)
    k = torch.randn(1, 1, 2, HEAD_DIM)
    v = torch.randn(1, 1, 2, HEAD_DIM)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.shape == k.shape
    assert torch.isfinite(k2).all() and torch.isfinite(v2).all()


def test_edge_case_head_dim_equals_group_size():
    # D == group_size => exactly one K group per (b,h,s).
    q = KIVIQuantizer(N_HEADS, head_dim=GROUP_SIZE, group_size=GROUP_SIZE)
    k, v = _rand_kv(B=2, H=N_HEADS, S=32, D=GROUP_SIZE)
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert k2.shape == k.shape
    assert torch.allclose(k, k2, atol=0.15)


# ---------------------------------------------------------------------------
# 6. determinism
# ---------------------------------------------------------------------------


def test_determinism():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(seed=42)
    c1 = q.compress(k, v)
    c2 = q.compress(k, v)
    assert torch.equal(c1.k_q, c2.k_q)
    assert torch.equal(c1.v_q, c2.v_q)
    assert torch.equal(c1.k_scale, c2.k_scale)
    assert torch.equal(c1.v_zero, c2.v_zero)


# ---------------------------------------------------------------------------
# 7. append extends compressed state
# ---------------------------------------------------------------------------


def test_append_extends_state():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k1, v1 = _rand_kv(B=2, S=32, seed=1)
    k2, v2 = _rand_kv(B=2, S=16, seed=2)
    c = q.compress(k1, v1)
    c2 = q.append(c, k2, v2)
    assert c2.shape == (2, N_HEADS, 48, HEAD_DIM)
    k_full, v_full = q.decompress(c2)
    assert k_full.shape == (2, N_HEADS, 48, HEAD_DIM)
    # The history portion should round-trip within the INT4 error band.
    # Append triggers a re-quantize, so V's per-token groups see the
    # combined span; allow a slightly looser atol than the cold compress.
    k1_rt = k_full[:, :, :32, :]
    assert torch.allclose(k1, k1_rt, atol=0.2)


# ---------------------------------------------------------------------------
# 8. no NaN/Inf
# ---------------------------------------------------------------------------


def test_no_nan_inf():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=2, S=64, seed=11)
    # Add a few extreme magnitudes to exercise scale edges.
    k[0, 0, 0, 0] = 1e4
    v[0, 0, 0, 0] = -1e4
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    assert torch.isfinite(k2).all()
    assert torch.isfinite(v2).all()
    assert torch.isfinite(c.k_scale).all()
    assert torch.isfinite(c.v_scale).all()


# ---------------------------------------------------------------------------
# 9. scales positive, zero_points (i.e., quantized codes) in [0, 15]
# ---------------------------------------------------------------------------


def test_scales_positive_and_codes_in_range():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=2, S=32, seed=3)
    c = q.compress(k, v)
    assert (c.k_scale > 0).all()
    assert (c.v_scale > 0).all()
    # Unpacked codes must live in [0, 15].
    k_codes = unpack_int4(c.k_q)
    v_codes = unpack_int4(c.v_q)
    assert int(k_codes.min()) >= 0 and int(k_codes.max()) <= 15
    assert int(v_codes.min()) >= 0 and int(v_codes.max()) <= 15


# ---------------------------------------------------------------------------
# 10. pack/unpack round-trip (helpers)
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip():
    g = torch.Generator().manual_seed(123)
    x = torch.randint(0, 16, (2, 3, 4, 8), generator=g, dtype=torch.int8)
    packed = pack_int4(x)
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == x.shape[-1] // 2
    restored = unpack_int4(packed)
    assert restored.shape == x.shape
    assert torch.equal(restored.to(torch.int8), x)


# ---------------------------------------------------------------------------
# 11. pack_int4 rejects out-of-range inputs
# ---------------------------------------------------------------------------


def test_pack_int4_rejects_out_of_range():
    bad_high = torch.tensor([[0, 16]], dtype=torch.int8)
    with pytest.raises(ValueError):
        pack_int4(bad_high)
    bad_low = torch.tensor([[-1, 0]], dtype=torch.int8)
    with pytest.raises(ValueError):
        pack_int4(bad_low)
    odd = torch.zeros(5, dtype=torch.int8)
    with pytest.raises(ValueError):
        pack_int4(odd)


# ---------------------------------------------------------------------------
# 12. 1024-token sequence compresses+decompresses in <1s
# ---------------------------------------------------------------------------


def test_large_sequence_performance():
    q = KIVIQuantizer(N_HEADS, HEAD_DIM, GROUP_SIZE)
    k, v = _rand_kv(B=1, S=1024)
    t0 = time.perf_counter()
    c = q.compress(k, v)
    k2, v2 = q.decompress(c)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"compress+decompress took {elapsed:.3f}s"
    assert k2.shape == k.shape


# ---------------------------------------------------------------------------
# Extra: validation — group_size must divide head_dim
# ---------------------------------------------------------------------------


def test_group_size_must_divide_head_dim():
    with pytest.raises(ValueError):
        KIVIQuantizer(n_heads=2, head_dim=16, group_size=6)
