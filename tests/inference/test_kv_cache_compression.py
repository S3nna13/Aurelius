"""16 tests for KV cache compression with token eviction."""

from __future__ import annotations

import torch

from src.inference.kv_cache_compression import (
    CompressedKVCache,
    KVCache,
    KVCacheConfig,
    evict_attention_sink,
    evict_heavy_hitter,
    evict_recent_only,
)

# ---------------------------------------------------------------------------
# Shared dimensions (as specified)
# ---------------------------------------------------------------------------

N_LAYERS = 2
N_HEADS = 2
HEAD_DIM = 8
MAX_CACHE = 10
SINK = 2
B = 1


def make_config(**kwargs) -> KVCacheConfig:
    defaults = dict(
        max_cache_size=MAX_CACHE,
        eviction_strategy="attention_sink",
        sink_tokens=SINK,
        recent_tokens=MAX_CACHE - SINK,
        heavy_hitter_ratio=0.2,
    )
    defaults.update(kwargs)
    return KVCacheConfig(**defaults)


def rand_kv(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)
    v = torch.randn(B, N_HEADS, T, HEAD_DIM)
    return k, v


# ---------------------------------------------------------------------------
# Test 1: KVCacheConfig defaults
# ---------------------------------------------------------------------------


def test_kvcacheconfig_defaults():
    cfg = KVCacheConfig()
    assert cfg.max_cache_size == 512
    assert cfg.eviction_strategy == "attention_sink"
    assert cfg.sink_tokens == 4
    assert cfg.recent_tokens == 256
    assert cfg.heavy_hitter_ratio == 0.2


# ---------------------------------------------------------------------------
# Test 2: KVCache update returns correct shape
# ---------------------------------------------------------------------------


def test_kvcache_update_returns_correct_shape():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)
    k, v = rand_kv(5)
    out_k, out_v = cache.update(0, k, v)
    assert out_k.shape == (B, N_HEADS, 5, HEAD_DIM)
    assert out_v.shape == (B, N_HEADS, 5, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test 3: KVCache update appends correctly (small cache, no eviction)
# ---------------------------------------------------------------------------


def test_kvcache_update_appends_without_eviction():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)
    k1, v1 = rand_kv(3)
    k2, v2 = rand_kv(4)
    cache.update(0, k1, v1)
    out_k, out_v = cache.update(0, k2, v2)
    # 3 + 4 = 7 which is < MAX_CACHE=10, so no eviction
    assert out_k.shape[2] == 7
    assert out_v.shape[2] == 7


# ---------------------------------------------------------------------------
# Test 4: KVCache get returns None before update
# ---------------------------------------------------------------------------


def test_kvcache_get_returns_none_before_update():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)
    result = cache.get(0)
    assert result is None
    result = cache.get(1)
    assert result is None


# ---------------------------------------------------------------------------
# Test 5: KVCache clear resets to empty
# ---------------------------------------------------------------------------


def test_kvcache_clear_resets():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)
    k, v = rand_kv(5)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert len(cache) == 5
    cache.clear()
    assert len(cache) == 0
    assert cache.get(0) is None
    assert cache.get(1) is None


# ---------------------------------------------------------------------------
# Test 6: KVCache __len__ correct
# ---------------------------------------------------------------------------


def test_kvcache_len_correct():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)
    assert len(cache) == 0
    k, v = rand_kv(6)
    cache.update(0, k, v)
    assert len(cache) == 6


# ---------------------------------------------------------------------------
# Test 7: evict_attention_sink keeps sink_tokens at start
# ---------------------------------------------------------------------------


def test_evict_attention_sink_keeps_sink_at_start():
    cfg = make_config(sink_tokens=SINK)
    T = 20  # > MAX_CACHE

    # Make each token position identifiable: token i has value float(i)
    keys = torch.stack([torch.full((B, N_HEADS, HEAD_DIM), float(i)) for i in range(T)], dim=2)
    values = keys.clone()

    out_k, out_v = evict_attention_sink(keys, values, cfg)

    # First SINK slots must be the original sink tokens (value 0.0 and 1.0)
    for i in range(SINK):
        expected = float(i)
        assert torch.allclose(out_k[:, :, i, :], torch.full_like(out_k[:, :, i, :], expected)), (
            f"Sink token {i} not preserved; got {out_k[:, :, i, :].unique()}"
        )


# ---------------------------------------------------------------------------
# Test 8: evict_attention_sink output length <= max_cache_size
# ---------------------------------------------------------------------------


def test_evict_attention_sink_output_length():
    cfg = make_config()
    T = 25
    k, v = rand_kv(T)
    out_k, out_v = evict_attention_sink(k, v, cfg)
    assert out_k.shape[2] <= MAX_CACHE
    assert out_v.shape[2] <= MAX_CACHE


# ---------------------------------------------------------------------------
# Test 9: evict_recent_only keeps only tail
# ---------------------------------------------------------------------------


def test_evict_recent_only_keeps_tail():
    cfg = make_config()
    T = 20
    # Token i has value float(i+1)
    keys = torch.stack([torch.full((B, N_HEADS, HEAD_DIM), float(i + 1)) for i in range(T)], dim=2)
    values = keys.clone()

    out_k, out_v = evict_recent_only(keys, values, cfg)

    assert out_k.shape[2] == MAX_CACHE
    # Last token in output should match last token of input
    last_input_val = float(T)  # token index T-1 has value float(T-1+1) = float(T)
    assert torch.allclose(out_k[:, :, -1, :], torch.full_like(out_k[:, :, -1, :], last_input_val))


# ---------------------------------------------------------------------------
# Test 10: evict_heavy_hitter with None attention falls back to recent
# ---------------------------------------------------------------------------


def test_evict_heavy_hitter_none_attention_falls_back_to_recent():
    cfg = make_config(eviction_strategy="heavy_hitter")
    T = 20
    k, v = rand_kv(T)

    out_k_hh, out_v_hh = evict_heavy_hitter(k, v, None, cfg)
    out_k_re, out_v_re = evict_recent_only(k, v, cfg)

    assert torch.allclose(out_k_hh, out_k_re)
    assert torch.allclose(out_v_hh, out_v_re)


# ---------------------------------------------------------------------------
# Test 11: evict_heavy_hitter with attention weights reduces cache
# ---------------------------------------------------------------------------


def test_evict_heavy_hitter_with_attention_reduces_cache():
    cfg = make_config()
    T = 20
    k, v = rand_kv(T)
    # attention_weights shape: (B, n_heads, T)
    attn_w = torch.rand(B, N_HEADS, T)

    out_k, out_v = evict_heavy_hitter(k, v, attn_w, cfg)
    assert out_k.shape[2] <= MAX_CACHE
    assert out_v.shape[2] <= MAX_CACHE


# ---------------------------------------------------------------------------
# Test 12: CompressedKVCache stats tracks evictions
# ---------------------------------------------------------------------------


def test_compressed_kvcache_stats_tracks_evictions():
    cfg = make_config()
    cache = CompressedKVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)

    # Add tokens that exceed max_cache_size to trigger eviction
    k, v = rand_kv(MAX_CACHE + 5)  # 15 > 10
    cache.update(0, k, v)

    stats = cache.get_stats()
    assert stats.n_evictions >= 1
    assert stats.total_tokens_evicted > 0


# ---------------------------------------------------------------------------
# Test 13: CompressedKVCache compression_ratio > 0 after eviction
# ---------------------------------------------------------------------------


def test_compressed_kvcache_compression_ratio_after_eviction():
    cfg = make_config()
    cache = CompressedKVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)

    k, v = rand_kv(MAX_CACHE + 5)
    cache.update(0, k, v)

    stats = cache.get_stats()
    assert stats.compression_ratio > 0.0
    assert stats.compression_ratio <= 1.0


# ---------------------------------------------------------------------------
# Test 14: KVCache update triggers eviction when over max_cache_size
# ---------------------------------------------------------------------------


def test_kvcache_update_triggers_eviction_over_max():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)

    # Push well over max_cache_size
    k, v = rand_kv(MAX_CACHE + 8)  # 18 > 10
    out_k, out_v = cache.update(0, k, v)

    assert out_k.shape[2] <= MAX_CACHE
    assert out_v.shape[2] <= MAX_CACHE


# ---------------------------------------------------------------------------
# Test 15: KVCache update with multiple layers independently
# ---------------------------------------------------------------------------


def test_kvcache_multiple_layers_independent():
    cfg = make_config()
    cache = KVCache(cfg, N_LAYERS, N_HEADS, HEAD_DIM)

    k0, v0 = rand_kv(5)
    k1, v1 = rand_kv(7)

    cache.update(0, k0, v0)
    cache.update(1, k1, v1)

    res0 = cache.get(0)
    res1 = cache.get(1)

    assert res0 is not None
    assert res1 is not None
    assert res0[0].shape[2] == 5
    assert res1[0].shape[2] == 7


# ---------------------------------------------------------------------------
# Test 16: evict_attention_sink preserves sink tokens exactly
# ---------------------------------------------------------------------------


def test_evict_attention_sink_preserves_sink_tokens_exactly():
    cfg = make_config(sink_tokens=SINK)
    T = 18

    # Each token has a distinct, known value
    keys = torch.zeros(B, N_HEADS, T, HEAD_DIM)
    for i in range(T):
        keys[:, :, i, :] = float(i * 10 + 1)
    values = keys.clone()

    out_k, out_v = evict_attention_sink(keys, values, cfg)

    # Verify exact values of the first SINK tokens are unchanged
    for i in range(SINK):
        expected_val = float(i * 10 + 1)
        assert torch.allclose(
            out_k[:, :, i, :],
            torch.full_like(out_k[:, :, i, :], expected_val),
        ), f"Sink token {i}: expected {expected_val}, got {out_k[:, :, i, :].unique()}"

    # Also check output length
    assert out_k.shape[2] == MAX_CACHE
