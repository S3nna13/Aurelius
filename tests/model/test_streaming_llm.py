"""Tests for StreamingLLM attention sink KV cache."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.streaming_llm import (
    KVCache,
    SinkAttentionWrapper,
    StreamingConfig,
    StreamingGenerator,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def streaming_cfg() -> StreamingConfig:
    return StreamingConfig(sink_size=2, window_size=4)


@pytest.fixture
def small_cache(streaming_cfg) -> KVCache:
    """Cache with sink=2, window=4, max=6, n_layers=2, n_heads=2, head_dim=16."""
    return KVCache(
        config=streaming_cfg,
        n_layers=2,
        n_heads=2,
        head_dim=16,
    )


def _make_token(n_heads: int = 2, head_dim: int = 16) -> torch.Tensor:
    """Return a single-token kv tensor: (1, n_heads, 1, head_dim)."""
    return torch.randn(1, n_heads, 1, head_dim)


# ---------------------------------------------------------------------------
# 1. test_kv_cache_appends_below_capacity
# ---------------------------------------------------------------------------


def test_kv_cache_appends_below_capacity():
    """Add 3 tokens to a capacity-5 cache; cache_size should be 3."""
    cfg = StreamingConfig(sink_size=2, window_size=3)  # max = 5
    cache = KVCache(cfg, n_layers=1, n_heads=2, head_dim=16)

    for _ in range(3):
        k, v = _make_token(), _make_token()
        cache.update(0, k, v)

    assert cache.cache_size == 3


# ---------------------------------------------------------------------------
# 2. test_kv_cache_evicts_oldest_recent
# ---------------------------------------------------------------------------


def test_kv_cache_evicts_oldest_recent(small_cache):
    """At capacity=6 (sink=2, window=4), adding a 7th token keeps 6 total."""
    # Fill to capacity.
    for _ in range(6):
        small_cache.update(0, _make_token(), _make_token())

    assert small_cache.cache_size == 6

    # One more token triggers eviction.
    small_cache.update(0, _make_token(), _make_token())

    # Still at max capacity.
    assert small_cache.cache_size == 6


# ---------------------------------------------------------------------------
# 3. test_kv_cache_sink_tokens_preserved
# ---------------------------------------------------------------------------


def test_kv_cache_sink_tokens_preserved():
    """After overflow, the first sink_size tokens must remain unchanged."""
    sink = 2
    cfg = StreamingConfig(sink_size=sink, window_size=3)  # max=5
    cache = KVCache(cfg, n_layers=1, n_heads=2, head_dim=16)

    sentinel = torch.ones(1, 2, 1, 16) * 99.0

    # Insert two sentinel sink tokens.
    cache.update(0, sentinel.clone(), sentinel.clone())
    cache.update(0, sentinel.clone(), sentinel.clone())

    # Fill the rest to capacity and then overflow.
    for _ in range(4):
        cache.update(0, _make_token(), _make_token())

    cached_k, _ = cache.get(0)
    # First `sink` tokens should still be the sentinel value.
    assert cached_k is not None
    assert torch.allclose(cached_k[:, :, :sink, :], sentinel.expand(1, 2, sink, 16))


# ---------------------------------------------------------------------------
# 4. test_kv_cache_clear_single_layer
# ---------------------------------------------------------------------------


def test_kv_cache_clear_single_layer(small_cache):
    """clear(layer_idx=0) empties layer 0 but leaves layer 1 intact."""
    # Populate both layers.
    small_cache.update(0, _make_token(), _make_token())
    small_cache.update(1, _make_token(), _make_token())

    small_cache.clear(layer_idx=0)

    k0, v0 = small_cache.get(0)
    k1, v1 = small_cache.get(1)

    assert k0 is None
    assert v0 is None
    assert k1 is not None
    assert v1 is not None


# ---------------------------------------------------------------------------
# 5. test_kv_cache_clear_all
# ---------------------------------------------------------------------------


def test_kv_cache_clear_all(small_cache):
    """clear() with no argument empties all layers."""
    small_cache.update(0, _make_token(), _make_token())
    small_cache.update(1, _make_token(), _make_token())

    small_cache.clear()

    for layer_idx in range(2):
        k, v = small_cache.get(layer_idx)
        assert k is None
        assert v is None


# ---------------------------------------------------------------------------
# 6. test_kv_cache_get_returns_none_when_empty
# ---------------------------------------------------------------------------


def test_kv_cache_get_returns_none_when_empty():
    """get() before any update must return (None, None)."""
    cfg = StreamingConfig()
    cache = KVCache(cfg, n_layers=2, n_heads=4, head_dim=16)

    k, v = cache.get(0)
    assert k is None
    assert v is None

    k, v = cache.get(1)
    assert k is None
    assert v is None


# ---------------------------------------------------------------------------
# 7. test_sink_attention_wrapper_no_cache_passthrough
# ---------------------------------------------------------------------------


def test_sink_attention_wrapper_no_cache_passthrough(small_cfg):
    """With cache=None, SinkAttentionWrapper calls the underlying attention."""
    from src.model.attention import GroupedQueryAttention, precompute_rope_frequencies

    attn = GroupedQueryAttention(small_cfg)
    cfg = StreamingConfig()
    wrapper = SinkAttentionWrapper(attn, cfg)

    x = torch.randn(1, 8, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 8)

    # cache=None -> prefill passthrough
    out = wrapper(x, cache=None, freqs_cis=freqs)
    assert out.shape == (1, 8, small_cfg.d_model)


# ---------------------------------------------------------------------------
# 8. test_streaming_generator_stream_returns_tokens
# ---------------------------------------------------------------------------


def test_streaming_generator_stream_returns_tokens(small_cfg):
    """stream() should return a list of the requested number of token ids."""
    model = AureliusTransformer(small_cfg)
    model.train(False)

    # Use a non-existent EOS id so generation always runs to max_tokens.
    gen = StreamingGenerator(
        model,
        config=StreamingConfig(sink_size=2, window_size=8),
        eos_token_id=9999,
    )

    prompt = [1, 2, 3, 4]
    max_tokens = 5
    tokens = gen.stream(prompt, max_tokens=max_tokens)

    assert isinstance(tokens, list)
    assert len(tokens) == max_tokens


# ---------------------------------------------------------------------------
# 9. test_streaming_config_max_cache_size
# ---------------------------------------------------------------------------


def test_streaming_config_max_cache_size():
    """max_cache_size should equal sink_size + window_size."""
    cfg = StreamingConfig(sink_size=4, window_size=512)
    assert cfg.max_cache_size == cfg.sink_size + cfg.window_size

    cfg2 = StreamingConfig(sink_size=8, window_size=256)
    assert cfg2.max_cache_size == cfg2.sink_size + cfg2.window_size


# ---------------------------------------------------------------------------
# 10. test_kv_cache_layer_independence
# ---------------------------------------------------------------------------


def test_kv_cache_layer_independence():
    """Updating layer 0 and layer 1 independently should not interfere."""
    cfg = StreamingConfig(sink_size=2, window_size=4)
    cache = KVCache(cfg, n_layers=2, n_heads=2, head_dim=16)

    # Insert 3 tokens into layer 0 and 5 tokens into layer 1.
    for _ in range(3):
        cache.update(0, _make_token(), _make_token())
    for _ in range(5):
        cache.update(1, _make_token(), _make_token())

    k0, _ = cache.get(0)
    k1, _ = cache.get(1)

    assert k0 is not None and k0.shape[2] == 3
    # Layer 1 inserted 5 tokens, which is below max_cache_size=6, so 5 total.
    assert k1 is not None and k1.shape[2] == 5
