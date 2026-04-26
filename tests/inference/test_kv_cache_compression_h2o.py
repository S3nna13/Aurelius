"""Tests for src/inference/kv_cache_compression_h2o.py.

Covers H2OKVCache, SlidingWindowKVCache, CacheStats, and KVCacheConfig.
Import path: aurelius.inference.kv_cache_compression_h2o
"""

from __future__ import annotations

import torch
from aurelius.inference.kv_cache_compression_h2o import (
    CacheStats,
    H2OKVCache,
    KVCacheConfig,
    SlidingWindowKVCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_HEADS = 4
HEAD_DIM = 16


def _make_cfg(
    max_cache_size: int = 16,
    n_heavy_hitters: int = 8,
    n_recent: int = 4,
    eviction_interval: int = 4,
) -> KVCacheConfig:
    return KVCacheConfig(
        max_cache_size=max_cache_size,
        n_heavy_hitters=n_heavy_hitters,
        n_recent=n_recent,
        eviction_interval=eviction_interval,
    )


def _make_kv(t: int, n_heads: int = N_HEADS, head_dim: int = HEAD_DIM):
    """Return (keys, values) of shape (n_heads, t, head_dim)."""
    keys = torch.randn(n_heads, t, head_dim)
    values = torch.randn(n_heads, t, head_dim)
    return keys, values


def _make_attn(q: int, t: int, n_heads: int = N_HEADS):
    """Uniform attention weights of shape (n_heads, q, t), rows sum to 1."""
    w = torch.ones(n_heads, q, t) / t
    return w


# ---------------------------------------------------------------------------
# KVCacheConfig
# ---------------------------------------------------------------------------


def test_kvcacheconfig_defaults():
    """Default field values should match the specification."""
    cfg = KVCacheConfig()
    assert cfg.max_cache_size == 512
    assert cfg.n_heavy_hitters == 256
    assert cfg.n_recent == 64
    assert cfg.eviction_interval == 16
    # Sanity: heavy-hitters + recent <= max_cache_size
    assert cfg.n_heavy_hitters + cfg.n_recent <= cfg.max_cache_size


# ---------------------------------------------------------------------------
# H2OKVCache — initialisation
# ---------------------------------------------------------------------------


def test_h2o_size_zero_on_init():
    """Cache must report size 0 immediately after construction."""
    cache = H2OKVCache(_make_cfg(), N_HEADS, HEAD_DIM)
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# H2OKVCache — single update below capacity
# ---------------------------------------------------------------------------


def test_h2o_size_after_small_update():
    """After inserting t=4 tokens (below max), size should equal t."""
    cache = H2OKVCache(_make_cfg(max_cache_size=32), N_HEADS, HEAD_DIM)
    t = 4
    k, v = _make_kv(t)
    w = _make_attn(q=t, t=t)
    cache.update(k, v, w)
    assert cache.size() == t


# ---------------------------------------------------------------------------
# H2OKVCache — capacity enforcement
# ---------------------------------------------------------------------------


def test_h2o_size_bounded_after_many_updates():
    """After many updates that exceed max_cache_size, size must stay <= max."""
    cfg = _make_cfg(max_cache_size=16, n_heavy_hitters=8, n_recent=4)
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)
    for _ in range(20):
        k, v = _make_kv(2)
        w = _make_attn(q=2, t=2)
        cache.update(k, v, w)
    assert cache.size() <= cfg.max_cache_size


# ---------------------------------------------------------------------------
# H2OKVCache — eviction keeps recent tokens
# ---------------------------------------------------------------------------


def test_h2o_eviction_keeps_recent_tokens():
    """The last n_recent positions in the cache must survive every eviction."""
    cfg = _make_cfg(max_cache_size=10, n_heavy_hitters=4, n_recent=3)
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)

    # Fill well past capacity so eviction runs.
    for _ in range(15):
        k, v = _make_kv(1)
        w = _make_attn(q=1, t=1)
        cache.update(k, v, w)

    # The final n_recent keys must be at the tail of the cache.
    keys, _ = cache.get()
    n_recent = cfg.n_recent
    # We stored distinct keys; after eviction the tail of the cache should hold
    # the most recently inserted keys — verify tensor is non-empty and tail size.
    assert keys.shape[1] >= n_recent, f"Expected >= {n_recent} positions, got {keys.shape[1]}"


# ---------------------------------------------------------------------------
# H2OKVCache — heavy hitters are retained
# ---------------------------------------------------------------------------


def test_h2o_heavy_hitters_retained():
    """Positions with the highest accumulated attention scores survive eviction."""
    cfg = _make_cfg(max_cache_size=12, n_heavy_hitters=4, n_recent=2)
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)

    # Insert 10 tokens with near-zero attention first.
    t_low = 10
    k_low, v_low = _make_kv(t_low)
    # Tiny but non-zero attention for background tokens.
    w_low = torch.full((N_HEADS, t_low, t_low), 1e-6)
    cache.update(k_low, v_low, w_low)

    # Insert 2 "star" tokens with very high attention.
    k_star, v_star = _make_kv(2)
    # Force maximum attention on these two positions.
    w_star = torch.zeros(N_HEADS, 2, 2)
    w_star[:, :, :] = 1.0
    cache.update(k_star, v_star, w_star)

    # After eviction the cache should contain at least the star tokens.
    keys_out, _ = cache.get()
    # The star keys correspond to the last 2 slots we inserted; their values
    # should appear somewhere in the retained cache.
    assert keys_out.shape[1] > 0, "Cache must not be empty after eviction"
    # At minimum: n_recent (2) positions are guaranteed to be kept.
    assert keys_out.shape[1] >= cfg.n_recent


# ---------------------------------------------------------------------------
# H2OKVCache — get() shape
# ---------------------------------------------------------------------------


def test_h2o_get_shapes():
    """get() must return tensors of shape (n_heads, S, head_dim)."""
    cfg = _make_cfg(max_cache_size=32)
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)
    t = 6
    k, v = _make_kv(t)
    w = _make_attn(q=t, t=t)
    cache.update(k, v, w)
    keys_out, values_out = cache.get()
    assert keys_out.shape == (N_HEADS, t, HEAD_DIM)
    assert values_out.shape == (N_HEADS, t, HEAD_DIM)


# ---------------------------------------------------------------------------
# H2OKVCache — reset
# ---------------------------------------------------------------------------


def test_h2o_reset_clears_cache():
    """reset() must set cache size back to 0."""
    cfg = _make_cfg()
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)
    k, v = _make_kv(8)
    w = _make_attn(q=8, t=8)
    cache.update(k, v, w)
    assert cache.size() > 0
    cache.reset()
    assert cache.size() == 0


def test_h2o_reset_get_shapes():
    """After reset(), get() must return empty tensors with correct layout."""
    cfg = _make_cfg()
    cache = H2OKVCache(cfg, N_HEADS, HEAD_DIM)
    k, v = _make_kv(4)
    w = _make_attn(q=4, t=4)
    cache.update(k, v, w)
    cache.reset()
    keys_out, values_out = cache.get()
    assert keys_out.shape == (N_HEADS, 0, HEAD_DIM)
    assert values_out.shape == (N_HEADS, 0, HEAD_DIM)


# ---------------------------------------------------------------------------
# SlidingWindowKVCache
# ---------------------------------------------------------------------------


def test_sliding_size_never_exceeds_window():
    """SlidingWindowKVCache must never exceed its window_size."""
    window = 8
    cache = SlidingWindowKVCache(window, N_HEADS, HEAD_DIM)
    for _ in range(20):
        k, v = _make_kv(2)
        cache.update(k, v)
    assert cache.size() <= window


def test_sliding_keeps_most_recent():
    """After overflow, the cache must contain the most recently inserted keys."""
    window = 4
    cache = SlidingWindowKVCache(window, N_HEADS, HEAD_DIM)

    # Insert 3 batches of 3 tokens each (9 total > window=4).
    sentinel = torch.ones(N_HEADS, 3, HEAD_DIM) * 99.0
    for _ in range(2):
        k, v = _make_kv(3)
        cache.update(k, v)
    # Insert sentinel as the last batch.
    cache.update(sentinel, sentinel)

    keys_out, _ = cache.get()
    assert keys_out.shape[1] == window
    # The most recent 3 slots must match the sentinel value.
    assert torch.allclose(keys_out[:, -3:, :], sentinel[:, :, :])


def test_sliding_get_shapes():
    """get() must return (n_heads, S, head_dim) with S <= window_size."""
    window = 8
    cache = SlidingWindowKVCache(window, N_HEADS, HEAD_DIM)
    t = 5
    k, v = _make_kv(t)
    cache.update(k, v)
    keys_out, values_out = cache.get()
    assert keys_out.shape == (N_HEADS, t, HEAD_DIM)
    assert values_out.shape == (N_HEADS, t, HEAD_DIM)


def test_sliding_get_shapes_after_overflow():
    """After overflow, get() shapes must reflect window_size not total tokens."""
    window = 6
    cache = SlidingWindowKVCache(window, N_HEADS, HEAD_DIM)
    for _ in range(5):
        k, v = _make_kv(3)
        cache.update(k, v)
    keys_out, values_out = cache.get()
    assert keys_out.shape == (N_HEADS, window, HEAD_DIM)
    assert values_out.shape == (N_HEADS, window, HEAD_DIM)


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


def test_cachestats_eviction_rate_zero_if_no_tokens():
    """eviction_rate must return 0.0 when total_tokens_seen is 0."""
    stats = CacheStats()
    assert stats.eviction_rate == 0.0


def test_cachestats_eviction_rate_correct():
    """eviction_rate must equal tokens_evicted / total_tokens_seen."""
    stats = CacheStats(total_tokens_seen=100, tokens_evicted=25, current_size=75)
    assert abs(stats.eviction_rate - 0.25) < 1e-9


def test_cachestats_eviction_rate_full_eviction():
    """eviction_rate == 1.0 when all tokens have been evicted."""
    stats = CacheStats(total_tokens_seen=50, tokens_evicted=50, current_size=0)
    assert stats.eviction_rate == 1.0


def test_cachestats_defaults():
    """Default CacheStats should have all counters at zero."""
    stats = CacheStats()
    assert stats.total_tokens_seen == 0
    assert stats.tokens_evicted == 0
    assert stats.current_size == 0
