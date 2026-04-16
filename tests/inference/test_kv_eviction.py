"""
Tests for src/inference/kv_eviction.py

Tiny tensors: B=1, H=2, T=8, D_HEAD=4, MAX_CACHE=6, N_SINK=2, N_RECENT=2
"""

import pytest
import torch

from src.inference.kv_eviction import (
    EvictionConfig,
    HeavyHitterEviction,
    KVCacheManager,
    StreamingLLMEviction,
    compute_cache_hit_rate,
    estimate_cache_memory,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
B, H, T, D = 1, 2, 8, 4
MAX_CACHE = 6
N_SINK = 2
N_RECENT = 2


def make_cfg(**kwargs) -> EvictionConfig:
    defaults = dict(
        max_cache_size=MAX_CACHE,
        n_sink_tokens=N_SINK,
        n_recent_tokens=N_RECENT,
        eviction_strategy="heavy_hitter",
        score_decay=0.9,
    )
    defaults.update(kwargs)
    return EvictionConfig(**defaults)


def make_kv(t: int = T) -> tuple:
    keys = torch.randn(B, H, t, D)
    vals = torch.randn(B, H, t, D)
    return keys, vals


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = EvictionConfig()
    assert cfg.max_cache_size == 512
    assert cfg.n_sink_tokens == 4
    assert cfg.n_recent_tokens == 64
    assert cfg.eviction_strategy == "heavy_hitter"
    assert cfg.score_decay == 0.9


# ---------------------------------------------------------------------------
# 2. StreamingLLM — T <= max_cache_size → unchanged tensors returned
# ---------------------------------------------------------------------------
def test_streaming_llm_no_eviction_when_under_budget():
    cfg = make_cfg()
    eviction = StreamingLLMEviction(cfg)
    # T=6 == MAX_CACHE=6 → no eviction
    keys, vals = make_kv(t=MAX_CACHE)
    ek, ev = eviction(keys, vals)
    assert ek.shape == keys.shape
    assert ev.shape == vals.shape
    assert torch.equal(ek, keys)
    assert torch.equal(ev, vals)


# ---------------------------------------------------------------------------
# 3. StreamingLLM — eviction reduces T to n_sink + n_recent
# ---------------------------------------------------------------------------
def test_streaming_llm_eviction_reduces_length():
    cfg = make_cfg()
    eviction = StreamingLLMEviction(cfg)
    keys, vals = make_kv(t=T)  # T=8 > MAX_CACHE=6
    ek, ev = eviction(keys, vals)
    expected_len = N_SINK + N_RECENT
    assert ek.shape[2] == expected_len
    assert ev.shape[2] == expected_len


# ---------------------------------------------------------------------------
# 4. StreamingLLM — eviction keeps first n_sink tokens exactly
# ---------------------------------------------------------------------------
def test_streaming_llm_eviction_keeps_sink_tokens():
    cfg = make_cfg()
    eviction = StreamingLLMEviction(cfg)
    keys, vals = make_kv(t=T)
    ek, ev = eviction(keys, vals)
    # First N_SINK positions in output must match first N_SINK of input
    assert torch.equal(ek[:, :, :N_SINK, :], keys[:, :, :N_SINK, :])
    assert torch.equal(ev[:, :, :N_SINK, :], vals[:, :, :N_SINK, :])


# ---------------------------------------------------------------------------
# 5. StreamingLLM — eviction keeps last n_recent tokens exactly
# ---------------------------------------------------------------------------
def test_streaming_llm_eviction_keeps_recent_tokens():
    cfg = make_cfg()
    eviction = StreamingLLMEviction(cfg)
    keys, vals = make_kv(t=T)
    ek, ev = eviction(keys, vals)
    assert torch.equal(ek[:, :, N_SINK:, :], keys[:, :, -N_RECENT:, :])
    assert torch.equal(ev[:, :, N_SINK:, :], vals[:, :, -N_RECENT:, :])


# ---------------------------------------------------------------------------
# 6. HeavyHitter — update_scores runs without error
# ---------------------------------------------------------------------------
def test_heavy_hitter_update_scores_no_error():
    cfg = make_cfg()
    hh = HeavyHitterEviction(cfg)
    attn = torch.softmax(torch.randn(B, H, T, T), dim=-1)
    hh.update_scores(attn)  # must not raise
    assert hh.scores is not None
    assert hh.scores.shape == (T,)


# ---------------------------------------------------------------------------
# 7. HeavyHitter — evict output length <= max_cache_size
# ---------------------------------------------------------------------------
def test_heavy_hitter_evict_output_length():
    cfg = make_cfg()
    hh = HeavyHitterEviction(cfg)
    attn = torch.softmax(torch.randn(B, H, T, T), dim=-1)
    hh.update_scores(attn)
    keys, vals = make_kv(t=T)
    ek, ev = hh.evict(keys, vals)
    assert ek.shape[2] <= MAX_CACHE
    assert ev.shape[2] <= MAX_CACHE


# ---------------------------------------------------------------------------
# 8. HeavyHitter — evict always keeps first n_sink tokens (by position order)
# ---------------------------------------------------------------------------
def test_heavy_hitter_evict_keeps_sink_positions():
    cfg = make_cfg()
    hh = HeavyHitterEviction(cfg)
    # Give very low attention to sink positions to ensure score alone wouldn't keep them
    attn = torch.zeros(B, H, T, T)
    # Only last tokens receive attention
    attn[:, :, :, -1] = 1.0
    hh.update_scores(attn)
    keys, vals = make_kv(t=T)
    ek, ev = hh.evict(keys, vals)
    # The first N_SINK rows of keys must appear (in order) at the start of output
    assert torch.equal(ek[:, :, :N_SINK, :], keys[:, :, :N_SINK, :])
    assert torch.equal(ev[:, :, :N_SINK, :], vals[:, :, :N_SINK, :])


# ---------------------------------------------------------------------------
# 9. compute_cache_hit_rate — all requested are cached → 1.0
# ---------------------------------------------------------------------------
def test_cache_hit_rate_all_cached():
    requested = [0, 1, 2, 3, 4]
    cached = [0, 1, 2, 3, 4, 5, 6]
    assert compute_cache_hit_rate(requested, cached) == 1.0


# ---------------------------------------------------------------------------
# 10. compute_cache_hit_rate — none cached → 0.0
# ---------------------------------------------------------------------------
def test_cache_hit_rate_none_cached():
    requested = [10, 11, 12]
    cached = [0, 1, 2, 3]
    assert compute_cache_hit_rate(requested, cached) == 0.0


# ---------------------------------------------------------------------------
# 11. estimate_cache_memory — formula check
# ---------------------------------------------------------------------------
def test_estimate_cache_memory_formula():
    max_size, n_layers, n_heads, d_head, dtype_bytes = 512, 12, 16, 64, 2
    expected = 2 * max_size * n_layers * n_heads * d_head * dtype_bytes
    result = estimate_cache_memory(max_size, n_layers, n_heads, d_head, dtype_bytes)
    assert result == expected


# ---------------------------------------------------------------------------
# 12. KVCacheManager.update — returns tensors with correct shape
# ---------------------------------------------------------------------------
def test_kvcache_manager_update_returns_tensors():
    cfg = make_cfg()
    manager = KVCacheManager(cfg)
    keys, vals = make_kv(t=4)
    rk, rv = manager.update(keys, vals)
    assert isinstance(rk, torch.Tensor)
    assert isinstance(rv, torch.Tensor)
    assert rk.dim() == 4
    assert rv.dim() == 4


# ---------------------------------------------------------------------------
# 13. manager.size() <= max_cache_size after overflow
# ---------------------------------------------------------------------------
def test_kvcache_manager_size_bounded_after_overflow():
    cfg = make_cfg()
    manager = KVCacheManager(cfg)
    # Insert more tokens than budget in two steps
    keys1, vals1 = make_kv(t=5)
    manager.update(keys1, vals1)
    keys2, vals2 = make_kv(t=5)
    manager.update(keys2, vals2)
    # After 10 tokens inserted with budget=6, size must be <= 6
    assert manager.size() <= MAX_CACHE


# ---------------------------------------------------------------------------
# 14. manager.reset() sets size to 0
# ---------------------------------------------------------------------------
def test_kvcache_manager_reset_clears_cache():
    cfg = make_cfg()
    manager = KVCacheManager(cfg)
    keys, vals = make_kv(t=4)
    manager.update(keys, vals)
    assert manager.size() > 0
    manager.reset()
    assert manager.size() == 0


# ---------------------------------------------------------------------------
# 15. HeavyHitter — reset clears scores
# ---------------------------------------------------------------------------
def test_heavy_hitter_reset_clears_scores():
    cfg = make_cfg()
    hh = HeavyHitterEviction(cfg)
    attn = torch.softmax(torch.randn(B, H, T, T), dim=-1)
    hh.update_scores(attn)
    assert hh.scores is not None
    hh.reset()
    assert hh.scores is None


# ---------------------------------------------------------------------------
# 16. compute_cache_hit_rate — partial hit
# ---------------------------------------------------------------------------
def test_cache_hit_rate_partial():
    requested = [0, 1, 2, 3]
    cached = [0, 2]  # 2 out of 4
    rate = compute_cache_hit_rate(requested, cached)
    assert rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 17. StreamingLLM — T exactly equal to max_cache_size → unchanged
# ---------------------------------------------------------------------------
def test_streaming_llm_exactly_at_budget():
    cfg = make_cfg()
    eviction = StreamingLLMEviction(cfg)
    keys, vals = make_kv(t=MAX_CACHE)
    ek, ev = eviction(keys, vals)
    assert ek.shape[2] == MAX_CACHE
    assert torch.equal(ek, keys)


# ---------------------------------------------------------------------------
# 18. estimate_cache_memory — dtype_bytes=4 (fp32)
# ---------------------------------------------------------------------------
def test_estimate_cache_memory_fp32():
    result = estimate_cache_memory(256, 6, 8, 32, dtype_bytes=4)
    expected = 2 * 256 * 6 * 8 * 32 * 4
    assert result == expected


# ---------------------------------------------------------------------------
# 19. KVCacheManager sequential updates stay bounded
# ---------------------------------------------------------------------------
def test_kvcache_manager_sequential_updates():
    cfg = make_cfg()
    manager = KVCacheManager(cfg)
    for _ in range(5):
        k, v = make_kv(t=3)
        manager.update(k, v)
    assert manager.size() <= MAX_CACHE


# ---------------------------------------------------------------------------
# 20. HeavyHitter — evict without prior update_scores (no scores) works
# ---------------------------------------------------------------------------
def test_heavy_hitter_evict_without_prior_scores():
    cfg = make_cfg()
    hh = HeavyHitterEviction(cfg)
    keys, vals = make_kv(t=T)  # T=8 > MAX_CACHE=6
    # No update_scores called — should fall back gracefully
    ek, ev = hh.evict(keys, vals)
    assert ek.shape[2] <= MAX_CACHE
