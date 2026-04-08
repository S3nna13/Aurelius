"""Tests for KV cache compression: H2O and StreamingLLM-style eviction."""

from __future__ import annotations

import torch
import pytest

from src.inference.kv_cache_compression import (
    AttentionScoreAccumulator,
    CachedAttentionLayer,
    H2OKVCache,
    KVCacheConfig,
    compress_kv_cache,
)

# ---------------------------------------------------------------------------
# Shared small dimensions
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
BUDGET = 8
N_SINK = 2
T = 16
B = 1


# ---------------------------------------------------------------------------
# 1. KVCacheConfig defaults
# ---------------------------------------------------------------------------


def test_kv_cache_config_defaults():
    cfg = KVCacheConfig()
    assert cfg.budget == 128
    assert cfg.n_sink_tokens == 4
    assert cfg.strategy == "h2o"
    assert cfg.accumulation_steps == 1


# ---------------------------------------------------------------------------
# 2. AttentionScoreAccumulator: scores shape after update
# ---------------------------------------------------------------------------


def test_attention_score_accumulator_update_shape():
    torch.manual_seed(0)
    T_k = 16
    n_heads = N_HEADS
    T_q = 4
    acc = AttentionScoreAccumulator(budget=BUDGET, n_sink_tokens=N_SINK)
    attn_weights = torch.rand(n_heads, T_q, T_k)
    acc.update(attn_weights)
    assert acc.scores is not None
    assert acc.scores.shape == (T_k,)


# ---------------------------------------------------------------------------
# 3. AttentionScoreAccumulator: heavy hitters are highest scored
# ---------------------------------------------------------------------------


def test_attention_score_accumulator_heavy_hitters():
    torch.manual_seed(0)
    T_k = 16
    acc = AttentionScoreAccumulator(budget=BUDGET, n_sink_tokens=N_SINK)
    # Create scores where we know which non-sink tokens are highest
    scores = torch.zeros(T_k)
    # Make tokens 5, 10 the highest non-sink
    scores[5] = 100.0
    scores[10] = 99.0
    acc.scores = scores

    n_to_fetch = 2
    hitters = acc.get_heavy_hitters(n_to_fetch)
    assert hitters.shape == (n_to_fetch,)
    # Results should be sorted ascending
    assert (hitters[1] > hitters[0]).item()
    # Both 5 and 10 should appear
    hitters_list = hitters.tolist()
    assert 5 in hitters_list
    assert 10 in hitters_list


# ---------------------------------------------------------------------------
# 4. H2OKVCache: size grows after update
# ---------------------------------------------------------------------------


def test_h2o_kvcache_update_appends():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK, strategy="h2o")
    cache = H2OKVCache(cfg, n_layers=1, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM)

    assert cache.size() == 0

    new_k = torch.randn(B, N_KV_HEADS, 4, HEAD_DIM)
    new_v = torch.randn(B, N_KV_HEADS, 4, HEAD_DIM)
    cache.update(0, new_k, new_v)
    assert cache.size() == 4

    new_k2 = torch.randn(B, N_KV_HEADS, 2, HEAD_DIM)
    new_v2 = torch.randn(B, N_KV_HEADS, 2, HEAD_DIM)
    cache.update(0, new_k2, new_v2)
    assert cache.size() == 6


# ---------------------------------------------------------------------------
# 5. H2OKVCache: size <= budget after many updates
# ---------------------------------------------------------------------------


def test_h2o_kvcache_evicts_when_full():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK, strategy="h2o")
    cache = H2OKVCache(cfg, n_layers=1, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM)

    for step in range(20):
        new_k = torch.randn(B, N_KV_HEADS, 1, HEAD_DIM)
        new_v = torch.randn(B, N_KV_HEADS, 1, HEAD_DIM)
        # Provide fake attention weights: (n_heads, 1, current_cache_size+1)
        T_k = cache.size() + 1
        attn_w = torch.rand(N_HEADS, 1, T_k)
        cache.update(0, new_k, new_v, attn_weights=attn_w)

    assert cache.size() <= BUDGET


# ---------------------------------------------------------------------------
# 6. H2OKVCache: first n_sink_tokens always preserved
# ---------------------------------------------------------------------------


def test_h2o_kvcache_keeps_sinks():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK, strategy="recent")
    cache = H2OKVCache(cfg, n_layers=1, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM)

    # Fill with identifiable values: token i has all values == float(i)
    initial_k = torch.stack([torch.full((B, N_KV_HEADS, HEAD_DIM), float(i)) for i in range(T)], dim=2)
    initial_v = initial_k.clone()
    cache.update(0, initial_k, initial_v)

    k, _ = cache.get(0)
    # First N_SINK tokens should be preserved (value 0.0 and 1.0)
    for sink_i in range(N_SINK):
        assert torch.allclose(k[:, :, sink_i, :], torch.full_like(k[:, :, sink_i, :], float(sink_i)))


# ---------------------------------------------------------------------------
# 7. H2OKVCache: reset clears cache, size == 0
# ---------------------------------------------------------------------------


def test_h2o_kvcache_reset():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK, strategy="h2o")
    cache = H2OKVCache(cfg, n_layers=1, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM)

    new_k = torch.randn(B, N_KV_HEADS, 4, HEAD_DIM)
    new_v = torch.randn(B, N_KV_HEADS, 4, HEAD_DIM)
    cache.update(0, new_k, new_v)
    assert cache.size() > 0

    cache.reset()
    assert cache.size() == 0
    assert cache.keys[0] is None
    assert cache.values[0] is None


# ---------------------------------------------------------------------------
# 8. compress_kv_cache: output shape (B, H, budget, D)
# ---------------------------------------------------------------------------


def test_compress_kv_cache_shape():
    torch.manual_seed(0)
    T_long = 32
    keys = torch.randn(B, N_HEADS, T_long, HEAD_DIM)
    values = torch.randn(B, N_HEADS, T_long, HEAD_DIM)
    scores = torch.rand(T_long)

    ck, cv = compress_kv_cache(keys, values, scores, budget=BUDGET, n_sink=N_SINK)
    assert ck.shape == (B, N_HEADS, BUDGET, HEAD_DIM)
    assert cv.shape == (B, N_HEADS, BUDGET, HEAD_DIM)


# ---------------------------------------------------------------------------
# 9. compress_kv_cache: first n_sink tokens always in output
# ---------------------------------------------------------------------------


def test_compress_kv_cache_keeps_sinks():
    torch.manual_seed(0)
    T_long = 32
    # Make sink tokens identifiable with large unique values
    keys = torch.randn(B, N_HEADS, T_long, HEAD_DIM)
    for i in range(N_SINK):
        keys[:, :, i, :] = float(i + 1) * 1000.0

    values = keys.clone()
    scores = torch.rand(T_long)

    ck, cv = compress_kv_cache(keys, values, scores, budget=BUDGET, n_sink=N_SINK)

    # The first N_SINK slots of the compressed output should match the original sink tokens
    for i in range(N_SINK):
        expected = float(i + 1) * 1000.0
        assert torch.allclose(ck[:, :, i, :], torch.full_like(ck[:, :, i, :], expected))


# ---------------------------------------------------------------------------
# 10. CachedAttentionLayer: forward output shape (B, T, D)
# ---------------------------------------------------------------------------


def test_cached_attention_layer_forward_shape():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK)
    layer = CachedAttentionLayer(D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x, use_cache=False)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 11. CachedAttentionLayer: use_cache=True works without error
# ---------------------------------------------------------------------------


def test_cached_attention_layer_with_cache():
    torch.manual_seed(0)
    cfg = KVCacheConfig(budget=BUDGET, n_sink_tokens=N_SINK, strategy="recent")
    layer = CachedAttentionLayer(D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, cfg)

    # Simulate decode: feed tokens one at a time (or small chunks)
    x_prefill = torch.randn(B, 4, D_MODEL)
    out_prefill = layer(x_prefill, use_cache=True)
    assert out_prefill.shape == (B, 4, D_MODEL)

    # Continue decode with new tokens
    x_decode = torch.randn(B, 1, D_MODEL)
    out_decode = layer(x_decode, use_cache=True)
    assert out_decode.shape == (B, 1, D_MODEL)

    # Verify cache is non-empty
    assert layer.cache.size() > 0
