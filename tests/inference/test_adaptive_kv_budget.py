"""
Tests for src/inference/adaptive_kv_budget.py — Adaptive per-layer KV budget allocation.

Config: n_layers=4, n_heads=2, d_head=8, total_budget=64
"""

from __future__ import annotations

import torch

from src.inference.adaptive_kv_budget import (
    AdaptiveKVCacheManager,
    AttentionPatternStats,
    KVBudgetAllocator,
    LayerKVCache,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

N_LAYERS = 4
N_HEADS = 2
D_HEAD = 8
TOTAL_BUDGET = 64
MIN_BUDGET = 16

B = 2  # batch size
T = 10  # sequence length for KV tensors


def make_attn(b: int = B, h: int = N_HEADS, t_q: int = 4, t_k: int = T) -> torch.Tensor:
    """Uniform, properly normalised attention weights (B, H, T_q, T_k)."""
    a = torch.ones(b, h, t_q, t_k)
    return a / t_k


def make_kv(b: int = B, h: int = N_HEADS, t: int = T, d: int = D_HEAD) -> torch.Tensor:
    return torch.randn(b, h, t, d)


# ===========================================================================
# AttentionPatternStats
# ===========================================================================


class TestAttentionPatternStatsUpdateCount:
    """Test 1 — update() increments update_count."""

    def test_update_increments_count(self):
        stats = AttentionPatternStats(N_LAYERS, N_HEADS)
        assert stats.update_count == 0
        attn = make_attn()
        stats.update(0, attn)
        assert stats.update_count == 1
        stats.update(1, attn)
        assert stats.update_count == 2


class TestAttentionPatternStatsEntropyHistoryShape:
    """Test 2 — entropy_history has n_layers entries on construction."""

    def test_entropy_history_length(self):
        stats = AttentionPatternStats(N_LAYERS, N_HEADS)
        assert len(stats.entropy_history) == N_LAYERS


class TestAttentionPatternStatsMeanEntropyEmpty:
    """Test 3 — mean_entropy returns 0.0 for a layer with no updates."""

    def test_mean_entropy_empty(self):
        stats = AttentionPatternStats(N_LAYERS, N_HEADS)
        result = stats.mean_entropy(2)
        assert result == 0.0


class TestAttentionPatternStatsMeanEntropyPositive:
    """Test 4 — mean_entropy returns positive float after update."""

    def test_mean_entropy_positive_after_update(self):
        stats = AttentionPatternStats(N_LAYERS, N_HEADS)
        # Uniform attention has positive entropy
        attn = make_attn()
        stats.update(0, attn)
        result = stats.mean_entropy(0)
        assert isinstance(result, float)
        assert result > 0.0, f"Expected positive entropy, got {result}"


class TestAttentionPatternStatsEntropyProfile:
    """Test 5 — entropy_profile returns a list of n_layers floats."""

    def test_entropy_profile_length_and_type(self):
        stats = AttentionPatternStats(N_LAYERS, N_HEADS)
        attn = make_attn()
        for i in range(N_LAYERS):
            stats.update(i, attn)
        profile = stats.entropy_profile()
        assert isinstance(profile, list)
        assert len(profile) == N_LAYERS
        for val in profile:
            assert isinstance(val, float)


# ===========================================================================
# KVBudgetAllocator
# ===========================================================================


class TestKVBudgetAllocatorUniformShape:
    """Test 6 — allocate_uniform returns a list of length n_layers."""

    def test_uniform_allocation_length(self):
        alloc = KVBudgetAllocator(N_LAYERS, TOTAL_BUDGET, MIN_BUDGET)
        budgets = alloc.allocate_uniform()
        assert len(budgets) == N_LAYERS


class TestKVBudgetAllocatorUniformSum:
    """Test 7 — uniform allocation sums to total_budget."""

    def test_uniform_allocation_sum(self):
        alloc = KVBudgetAllocator(N_LAYERS, TOTAL_BUDGET, MIN_BUDGET)
        budgets = alloc.allocate_uniform()
        assert sum(budgets) == TOTAL_BUDGET, f"Expected sum={TOTAL_BUDGET}, got {sum(budgets)}"


class TestKVBudgetAllocatorByEntropyShape:
    """Test 8 — allocate_by_entropy returns a list of length n_layers."""

    def test_entropy_allocation_length(self):
        alloc = KVBudgetAllocator(N_LAYERS, TOTAL_BUDGET, MIN_BUDGET)
        profile = [1.0, 2.0, 0.5, 1.5]
        budgets = alloc.allocate_by_entropy(profile)
        assert len(budgets) == N_LAYERS


class TestKVBudgetAllocatorByEntropyMinimum:
    """Test 9 — allocate_by_entropy: each entry >= min_budget_per_layer."""

    def test_entropy_allocation_min_budget(self):
        alloc = KVBudgetAllocator(N_LAYERS, TOTAL_BUDGET, MIN_BUDGET)
        # Skewed profile — one very high, others near zero
        profile = [10.0, 0.01, 0.01, 0.01]
        budgets = alloc.allocate_by_entropy(profile)
        for i, b in enumerate(budgets):
            assert b >= MIN_BUDGET, (
                f"Layer {i} budget {b} is below min_budget_per_layer={MIN_BUDGET}"
            )


# ===========================================================================
# LayerKVCache
# ===========================================================================


class TestLayerKVCacheUpdateShape:
    """Test 10 — update returns (B, H, ≤budget, d_head) tensors."""

    def test_update_output_shape(self):
        budget = 16
        cache = LayerKVCache(budget, N_HEADS, D_HEAD)
        k = make_kv(t=6)
        v = make_kv(t=6)
        out_k, out_v = cache.update(k, v)
        assert out_k.dim() == 4
        assert out_k.shape[0] == B
        assert out_k.shape[1] == N_HEADS
        assert out_k.shape[2] <= budget
        assert out_k.shape[3] == D_HEAD
        assert out_k.shape == out_v.shape


class TestLayerKVCacheGrowsThenCaps:
    """Test 11 — cache grows up to budget then stays at budget."""

    def test_cache_grows_and_caps(self):
        budget = 12
        cache = LayerKVCache(budget, N_HEADS, D_HEAD)

        # First update: 6 tokens — cache should hold 6
        k1, v1 = make_kv(t=6), make_kv(t=6)
        out_k, _ = cache.update(k1, v1)
        assert out_k.shape[2] == 6, f"Expected 6, got {out_k.shape[2]}"

        # Second update: 8 more tokens — 6+8=14 > 12, capped at 12
        k2, v2 = make_kv(t=8), make_kv(t=8)
        out_k, _ = cache.update(k2, v2)
        assert out_k.shape[2] == budget, f"Expected {budget}, got {out_k.shape[2]}"

        # Third update: 4 more tokens — still capped at 12
        k3, v3 = make_kv(t=4), make_kv(t=4)
        out_k, _ = cache.update(k3, v3)
        assert out_k.shape[2] == budget


class TestLayerKVCacheSizeMethod:
    """Test 12 — size() returns the correct cached token count."""

    def test_size_correct(self):
        cache = LayerKVCache(20, N_HEADS, D_HEAD)
        assert cache.size() == 0
        k, v = make_kv(t=5), make_kv(t=5)
        cache.update(k, v)
        assert cache.size() == 5
        k2, v2 = make_kv(t=7), make_kv(t=7)
        cache.update(k2, v2)
        assert cache.size() == 12


class TestLayerKVCacheClear:
    """Test 13 — clear() resets the cache to empty."""

    def test_clear_resets_cache(self):
        cache = LayerKVCache(20, N_HEADS, D_HEAD)
        k, v = make_kv(t=8), make_kv(t=8)
        cache.update(k, v)
        assert cache.size() == 8
        cache.clear()
        assert cache.size() == 0
        assert cache.keys is None
        assert cache.values is None


# ===========================================================================
# AdaptiveKVCacheManager
# ===========================================================================


class TestAdaptiveKVCacheManagerUpdateLayer:
    """Test 14 — update_layer returns a tensor pair."""

    def test_update_layer_returns_tensor_pair(self):
        mgr = AdaptiveKVCacheManager(N_LAYERS, N_HEADS, D_HEAD, TOTAL_BUDGET)
        k = make_kv(t=4)
        v = make_kv(t=4)
        out_k, out_v = mgr.update_layer(0, k, v)
        assert isinstance(out_k, torch.Tensor)
        assert isinstance(out_v, torch.Tensor)
        assert out_k.shape == out_v.shape
        # Shape: (B, H, ≤budget_layer_0, D_HEAD)
        assert out_k.dim() == 4
        assert out_k.shape[0] == B
        assert out_k.shape[1] == N_HEADS
        assert out_k.shape[3] == D_HEAD


class TestAdaptiveKVCacheManagerTotalCachedTokens:
    """Test 15 — total_cached_tokens returns a non-negative int."""

    def test_total_cached_tokens_non_negative(self):
        mgr = AdaptiveKVCacheManager(N_LAYERS, N_HEADS, D_HEAD, TOTAL_BUDGET)
        total = mgr.total_cached_tokens()
        assert isinstance(total, int)
        assert total >= 0

    def test_total_cached_tokens_increases_after_update(self):
        mgr = AdaptiveKVCacheManager(N_LAYERS, N_HEADS, D_HEAD, TOTAL_BUDGET)
        assert mgr.total_cached_tokens() == 0
        k, v = make_kv(t=4), make_kv(t=4)
        mgr.update_layer(0, k, v)
        mgr.update_layer(1, k, v)
        assert mgr.total_cached_tokens() == 8
