"""Tests for StreamingConfig, StreamingAttentionCache, and compute_streaming_attention."""

from __future__ import annotations

import pytest
import torch

from src.longcontext.streaming_attention import (
    StreamingConfig,
    StreamingAttentionCache,
    compute_streaming_attention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HEAD_DIM = 8

def _kv(dim: int = HEAD_DIM):
    k = torch.randn(dim)
    v = torch.randn(dim)
    return k, v

def _cache(**kw) -> StreamingAttentionCache:
    cfg = StreamingConfig(**kw)
    return StreamingAttentionCache(cfg)


# ===========================================================================
# StreamingConfig defaults
# ===========================================================================

def test_config_default_window_size():
    assert StreamingConfig().window_size == 512

def test_config_default_n_sink_tokens():
    assert StreamingConfig().n_sink_tokens == 4

def test_config_default_max_cache_size():
    assert StreamingConfig().max_cache_size == 520

def test_config_custom():
    cfg = StreamingConfig(window_size=32, n_sink_tokens=2, max_cache_size=34)
    assert cfg.window_size == 32
    assert cfg.n_sink_tokens == 2
    assert cfg.max_cache_size == 34


# ===========================================================================
# StreamingAttentionCache: empty state
# ===========================================================================

def test_empty_cache_len_is_zero():
    cache = _cache()
    assert len(cache) == 0

def test_empty_cache_sink_count():
    cache = _cache()
    assert cache.sink_count() == 0

def test_empty_cache_get_returns_none_none():
    cache = _cache()
    k, v = cache.get_cache()
    assert k is None
    assert v is None


# ===========================================================================
# add_token: length increases
# ===========================================================================

def test_add_token_len_increases():
    cache = _cache(max_cache_size=100)
    for i in range(5):
        k, v = _kv()
        cache.add_token(k, v)
        assert len(cache) == i + 1

def test_add_token_sink_count_increases_to_limit():
    cache = _cache(n_sink_tokens=3, max_cache_size=100)
    for i in range(3):
        cache.add_token(*_kv())
    assert cache.sink_count() == 3
    # Adding more tokens doesn't increase sink_count beyond n_sink_tokens
    cache.add_token(*_kv())
    assert cache.sink_count() == 3

def test_add_token_get_cache_returns_tensors():
    cache = _cache(max_cache_size=10)
    cache.add_token(*_kv())
    k, v = cache.get_cache()
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)

def test_add_token_get_cache_shape():
    cache = _cache(max_cache_size=10)
    n = 5
    for _ in range(n):
        cache.add_token(*_kv())
    k, v = cache.get_cache()
    assert k.shape[0] == n
    assert v.shape[0] == n


# ===========================================================================
# Eviction: oldest non-sink tokens removed
# ===========================================================================

def test_eviction_keeps_len_at_max():
    n_sink = 2
    max_size = 5
    cache = _cache(n_sink_tokens=n_sink, max_cache_size=max_size)
    for _ in range(max_size + 3):
        cache.add_token(*_kv())
    assert len(cache) == max_size

def test_eviction_preserves_sink_tokens():
    """Sink tokens must be the first n_sink items added."""
    n_sink = 2
    max_size = 4
    cache = _cache(n_sink_tokens=n_sink, max_cache_size=max_size)
    # Record sink keys
    sink_keys = []
    for i in range(n_sink):
        k = torch.full((HEAD_DIM,), float(i))
        v = torch.zeros(HEAD_DIM)
        cache.add_token(k, v)
        sink_keys.append(k)
    # Fill to trigger eviction
    for _ in range(5):
        cache.add_token(*_kv())
    cached_k, _ = cache.get_cache()
    # First n_sink rows should match sink_keys
    for i, sk in enumerate(sink_keys):
        assert torch.allclose(cached_k[i], sk), f"Sink {i} was evicted"

def test_eviction_removes_non_sink_oldest():
    """Non-sink tokens should be evicted in FIFO order."""
    n_sink = 1
    max_size = 3
    cache = _cache(n_sink_tokens=n_sink, max_cache_size=max_size)
    # Add sink
    sink_k = torch.full((HEAD_DIM,), 99.0)
    cache.add_token(sink_k, torch.zeros(HEAD_DIM))
    # Add two non-sink tokens (IDs 1, 2)
    k1 = torch.full((HEAD_DIM,), 1.0)
    k2 = torch.full((HEAD_DIM,), 2.0)
    cache.add_token(k1, torch.zeros(HEAD_DIM))
    cache.add_token(k2, torch.zeros(HEAD_DIM))
    assert len(cache) == 3
    # Add one more → k1 (oldest non-sink) should be evicted
    k3 = torch.full((HEAD_DIM,), 3.0)
    cache.add_token(k3, torch.zeros(HEAD_DIM))
    assert len(cache) == 3
    cached_k, _ = cache.get_cache()
    ids = [cached_k[i, 0].item() for i in range(3)]
    assert 99.0 in ids
    assert 1.0 not in ids
    assert 2.0 in ids
    assert 3.0 in ids


# ===========================================================================
# sink_count
# ===========================================================================

def test_sink_count_empty():
    cache = _cache(n_sink_tokens=4)
    assert cache.sink_count() == 0

def test_sink_count_partial():
    cache = _cache(n_sink_tokens=4, max_cache_size=100)
    cache.add_token(*_kv())
    cache.add_token(*_kv())
    assert cache.sink_count() == 2

def test_sink_count_full_sinks():
    cache = _cache(n_sink_tokens=4, max_cache_size=100)
    for _ in range(4):
        cache.add_token(*_kv())
    assert cache.sink_count() == 4

def test_sink_count_does_not_exceed_n_sink():
    cache = _cache(n_sink_tokens=4, max_cache_size=100)
    for _ in range(10):
        cache.add_token(*_kv())
    assert cache.sink_count() == 4


# ===========================================================================
# compute_streaming_attention: output shape
# ===========================================================================

def test_streaming_attn_empty_cache_returns_zeros():
    cache = _cache()
    q = torch.randn(1, 1, HEAD_DIM)
    out = compute_streaming_attention(q, cache)
    assert torch.allclose(out, torch.zeros_like(q))

def test_streaming_attn_empty_cache_shape():
    cache = _cache()
    q = torch.randn(1, 2, HEAD_DIM)
    out = compute_streaming_attention(q, cache)
    assert out.shape == q.shape

def test_streaming_attn_output_shape_3d():
    cache = _cache(max_cache_size=10)
    for _ in range(3):
        cache.add_token(*_kv())
    q = torch.randn(1, 1, HEAD_DIM)
    out = compute_streaming_attention(q, cache)
    assert out.shape == q.shape

def test_streaming_attn_output_shape_4d():
    cache = _cache(max_cache_size=10)
    for _ in range(3):
        cache.add_token(*_kv())
    q = torch.randn(1, 1, 1, HEAD_DIM)
    out = compute_streaming_attention(q, cache)
    assert out.shape == q.shape

def test_streaming_attn_output_is_tensor():
    cache = _cache(max_cache_size=10)
    cache.add_token(*_kv())
    q = torch.randn(1, 1, HEAD_DIM)
    out = compute_streaming_attention(q, cache)
    assert isinstance(out, torch.Tensor)

def test_streaming_attn_custom_scale():
    cache = _cache(max_cache_size=10)
    for _ in range(3):
        cache.add_token(*_kv())
    q = torch.randn(1, 1, HEAD_DIM)
    out = compute_streaming_attention(q, cache, scale=0.5)
    assert out.shape == q.shape

def test_streaming_attn_scale_affects_output():
    torch.manual_seed(0)
    cache1 = _cache(max_cache_size=10)
    cache2 = _cache(max_cache_size=10)
    for _ in range(3):
        k, v = _kv()
        cache1.add_token(k.clone(), v.clone())
        cache2.add_token(k.clone(), v.clone())
    q = torch.randn(1, 1, HEAD_DIM)
    out1 = compute_streaming_attention(q, cache1, scale=1.0)
    out2 = compute_streaming_attention(q, cache2, scale=0.1)
    assert not torch.allclose(out1, out2)

def test_streaming_attn_zeros_for_empty_cache_different_shapes():
    cache = _cache()
    for shape in [(2, 4, HEAD_DIM), (1, 1, HEAD_DIM), (3, 2, HEAD_DIM)]:
        q = torch.randn(*shape)
        out = compute_streaming_attention(q, cache)
        assert torch.allclose(out, torch.zeros_like(q))
