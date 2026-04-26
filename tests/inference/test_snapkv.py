"""
Tests for src/inference/snapkv.py — SnapKV KV-cache compression.

Config: B=2, n_heads=4, T=64, head_dim=16, window_size=8, max_capacity=32
"""

from __future__ import annotations

import torch

from src.inference.snapkv import SnapKVCache, SnapKVPolicy

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

B = 2
H = 4  # n_heads
T = 64  # seq_len
D = 16  # head_dim
W = 8  # window_size
K = 32  # max_capacity


def make_uniform_attn(b: int = 1, h: int = H, t_obs: int = W, t_seq: int = T) -> torch.Tensor:
    """Uniform attention weights (all equal)."""
    a = torch.ones(b, h, t_obs, t_seq)
    return a / t_seq


def make_peaked_attn(
    peak_positions: list[int],
    b: int = 1,
    h: int = H,
    t_obs: int = W,
    t_seq: int = T,
) -> torch.Tensor:
    """Attention heavily peaked at given positions, near-zero elsewhere."""
    a = torch.full((b, h, t_obs, t_seq), 1e-6)
    for pos in peak_positions:
        a[:, :, :, pos] = 1.0
    # Renormalise rows
    a = a / a.sum(dim=-1, keepdim=True)
    return a


# ---------------------------------------------------------------------------
# SnapKVPolicy tests
# ---------------------------------------------------------------------------


class TestSnapKVPolicyOutputLength:
    """Test 1 — select_indices returns ≤ max_capacity indices."""

    def test_output_length_le_max_capacity(self):
        policy = SnapKVPolicy(window_size=W, max_capacity=K)
        attn = make_uniform_attn()
        idx = policy.select_indices(attn, seq_len=T)
        assert idx.numel() <= K, f"Expected ≤ {K} indices, got {idx.numel()}"


class TestSnapKVPolicyObsWindowIncluded:
    """Test 2 — output always includes ALL observation-window indices."""

    def test_obs_window_always_included(self):
        policy = SnapKVPolicy(window_size=W, max_capacity=K)
        attn = make_uniform_attn()
        idx = policy.select_indices(attn, seq_len=T)
        idx_set = set(idx.tolist())
        expected_obs = set(range(T - W, T))
        assert expected_obs.issubset(idx_set), (
            f"Observation window {expected_obs} not fully in result {idx_set}"
        )


class TestSnapKVPolicySorted:
    """Test 3 — output indices are sorted ascending."""

    def test_indices_sorted_ascending(self):
        policy = SnapKVPolicy(window_size=W, max_capacity=K)
        attn = make_uniform_attn()
        idx = policy.select_indices(attn, seq_len=T)
        assert (idx[1:] > idx[:-1]).all(), "Indices are not strictly ascending"


class TestSnapKVPolicyHighImportanceSelected:
    """Test 4 — high-importance positions are selected."""

    def test_peak_positions_selected(self):
        # Peaks at positions 5 and 10 — well outside the obs window
        peaks = [5, 10]
        policy = SnapKVPolicy(window_size=W, max_capacity=K)
        attn = make_peaked_attn(peaks)
        idx = policy.select_indices(attn, seq_len=T)
        idx_set = set(idx.tolist())
        for p in peaks:
            assert p in idx_set, f"Peak position {p} not selected; got {idx_set}"


class TestSnapKVCacheShape:
    """Test 5 — compress returns (B, n_heads, k_kept, head_dim), k_kept ≤ max_capacity."""

    def test_output_shape(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_uniform_attn(b=B)
        ck, cv = cache.compress(keys, values, attn)
        assert ck.dim() == 4, f"Expected 4-D keys, got {ck.dim()}"
        assert ck.shape[0] == B
        assert ck.shape[1] == H
        assert ck.shape[2] <= K, f"k_kept={ck.shape[2]} exceeds max_capacity={K}"
        assert ck.shape[3] == D


class TestSnapKVCacheSameShape:
    """Test 6 — compressed keys and values have identical shape."""

    def test_keys_values_same_shape(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_uniform_attn(b=B)
        ck, cv = cache.compress(keys, values, attn)
        assert ck.shape == cv.shape, f"keys shape {ck.shape} != values shape {cv.shape}"


class TestSnapKVCacheDeterminism:
    """Test 7 — same attention weights produce identical results."""

    def test_determinism(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_uniform_attn(b=B)
        ck1, cv1 = cache.compress(keys, values, attn)
        ck2, cv2 = cache.compress(keys, values, attn)
        assert torch.equal(ck1, ck2), "Keys not deterministic under same inputs"
        assert torch.equal(cv1, cv2), "Values not deterministic under same inputs"


class TestSnapKVNoEviction:
    """Test 8 — when max_capacity ≥ T, no eviction occurs (all positions kept)."""

    def test_no_eviction_when_capacity_large(self):
        # max_capacity = T → nothing should be evicted
        policy = SnapKVPolicy(window_size=W, max_capacity=T)
        attn = make_uniform_attn()
        idx = policy.select_indices(attn, seq_len=T)
        assert idx.numel() == T, f"Expected all {T} indices kept, got {idx.numel()}"
        assert set(idx.tolist()) == set(range(T))


class TestSnapKVWindowClamping:
    """Test 9 — window_size > T is handled gracefully (clamped to T)."""

    def test_window_larger_than_seq(self):
        short_T = 10
        policy = SnapKVPolicy(window_size=50, max_capacity=32)
        # T_obs can be at most short_T; we pass a matching attn tensor
        attn = make_uniform_attn(t_obs=short_T, t_seq=short_T)
        # Should not raise; result ≤ min(short_T, 32)
        idx = policy.select_indices(attn, seq_len=short_T)
        assert idx.numel() <= short_T


class TestSnapKVMaxPool:
    """Test 10 — pool_method='max' works without crash."""

    def test_pool_max_no_crash(self):
        cache = SnapKVCache(window_size=W, max_capacity=K, pool_method="max")
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_uniform_attn(b=B)
        ck, cv = cache.compress(keys, values, attn)
        assert ck.shape[2] <= K


class TestSnapKVNoNaNUniform:
    """Test 11 — no NaN or Inf on uniform attention."""

    def test_no_nan_inf_uniform(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_uniform_attn(b=B)
        ck, cv = cache.compress(keys, values, attn)
        assert not torch.isnan(ck).any(), "NaN in compressed keys (uniform attn)"
        assert not torch.isnan(cv).any(), "NaN in compressed values (uniform attn)"
        assert not torch.isinf(ck).any(), "Inf in compressed keys (uniform attn)"
        assert not torch.isinf(cv).any(), "Inf in compressed values (uniform attn)"


class TestSnapKVNoNaNPeaked:
    """Test 12 — no NaN or Inf on peaked attention."""

    def test_no_nan_inf_peaked(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, H, T, D)
        values = torch.randn(B, H, T, D)
        attn = make_peaked_attn([3, 7, 15], b=B)
        ck, cv = cache.compress(keys, values, attn)
        assert not torch.isnan(ck).any(), "NaN in compressed keys (peaked attn)"
        assert not torch.isnan(cv).any(), "NaN in compressed values (peaked attn)"
        assert not torch.isinf(ck).any(), "Inf in compressed keys (peaked attn)"
        assert not torch.isinf(cv).any(), "Inf in compressed values (peaked attn)"


class TestSnapKVBatchSizeOne:
    """Test 13 — batch_size=1 works correctly."""

    def test_batch_size_one(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(1, H, T, D)
        values = torch.randn(1, H, T, D)
        attn = make_uniform_attn(b=1)
        ck, cv = cache.compress(keys, values, attn)
        assert ck.shape[0] == 1
        assert ck.shape[2] <= K


class TestSnapKVOneHead:
    """Test 14 — n_heads=1 works correctly."""

    def test_one_head(self):
        cache = SnapKVCache(window_size=W, max_capacity=K)
        keys = torch.randn(B, 1, T, D)
        values = torch.randn(B, 1, T, D)
        attn = make_uniform_attn(b=B, h=1)
        ck, cv = cache.compress(keys, values, attn)
        assert ck.shape[1] == 1
        assert ck.shape[2] <= K


class TestSnapKVTopKGuarantee:
    """Test 15 — highest-attended positions are always included in the result."""

    def test_highest_attended_always_included(self):
        # Create attention with clear top-3 positions outside obs window
        top_positions = [1, 20, 40]  # all well outside last W positions
        attn = make_peaked_attn(top_positions, b=1)
        policy = SnapKVPolicy(window_size=W, max_capacity=K)
        idx = policy.select_indices(attn, seq_len=T)
        idx_set = set(idx.tolist())
        for pos in top_positions:
            assert pos in idx_set, f"Top position {pos} not in selected indices {sorted(idx_set)}"
