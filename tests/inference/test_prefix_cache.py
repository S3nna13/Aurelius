"""Tests for src/inference/prefix_cache.py"""

from __future__ import annotations

import time

import pytest
import torch

from src.inference.prefix_cache import (
    PrefixCacheConfig,
    CacheEntry,
    PrefixCache,
    compute_prefix_hash,
    find_longest_prefix_match,
    truncate_kv_cache,
    merge_kv_caches,
    simulate_prefix_caching,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(n_layers: int = 2, n_heads: int = 2, seq_len: int = 4, head_dim: int = 16) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a fake KV cache with known shape (1, n_heads, seq_len, head_dim)."""
    return [
        (
            torch.randn(1, n_heads, seq_len, head_dim),
            torch.randn(1, n_heads, seq_len, head_dim),
        )
        for _ in range(n_layers)
    ]


def _make_entry(prefix_ids: list[int], hit_count: int = 0, ts: float | None = None) -> CacheEntry:
    now = ts if ts is not None else time.time()
    return CacheEntry(
        prefix_ids=prefix_ids,
        kv_cache=_make_kv(),
        hit_count=hit_count,
        last_accessed=now,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = PrefixCacheConfig()
    assert cfg.max_entries == 64
    assert cfg.max_prefix_len == 512
    assert cfg.min_prefix_len == 8
    assert cfg.eviction_policy == "lru"


# ---------------------------------------------------------------------------
# 2. test_compute_prefix_hash_deterministic
# ---------------------------------------------------------------------------

def test_compute_prefix_hash_deterministic():
    ids = [1, 2, 3, 4, 5]
    assert compute_prefix_hash(ids) == compute_prefix_hash(ids)


# ---------------------------------------------------------------------------
# 3. test_compute_prefix_hash_different
# ---------------------------------------------------------------------------

def test_compute_prefix_hash_different():
    assert compute_prefix_hash([1, 2, 3]) != compute_prefix_hash([1, 2, 4])
    assert compute_prefix_hash([1, 2, 3]) != compute_prefix_hash([3, 2, 1])


# ---------------------------------------------------------------------------
# 4. test_find_longest_prefix_empty_cache
# ---------------------------------------------------------------------------

def test_find_longest_prefix_empty_cache():
    key, length = find_longest_prefix_match([1, 2, 3], {})
    assert key is None
    assert length == 0


# ---------------------------------------------------------------------------
# 5. test_find_longest_prefix_exact
# ---------------------------------------------------------------------------

def test_find_longest_prefix_exact():
    prefix = [10, 20, 30]
    entry = _make_entry(prefix)
    cache_key = compute_prefix_hash(prefix)
    cache = {cache_key: entry}
    key, length = find_longest_prefix_match([10, 20, 30, 40], cache)
    assert key == cache_key
    assert length == 3


# ---------------------------------------------------------------------------
# 6. test_find_longest_prefix_partial
# ---------------------------------------------------------------------------

def test_find_longest_prefix_partial():
    short_prefix = [1, 2]
    long_prefix = [1, 2, 3, 4]
    entry_short = _make_entry(short_prefix)
    entry_long = _make_entry(long_prefix)

    key_short = compute_prefix_hash(short_prefix)
    key_long = compute_prefix_hash(long_prefix)

    cache = {key_short: entry_short, key_long: entry_long}

    # Query [1, 2, 3, 4, 5] should match long_prefix (length 4)
    key, length = find_longest_prefix_match([1, 2, 3, 4, 5], cache)
    assert length == 4
    assert key == key_long

    # Query [1, 2, 99] matches only short_prefix (length 2)
    key2, length2 = find_longest_prefix_match([1, 2, 99], cache)
    assert length2 == 2
    assert key2 == key_short


# ---------------------------------------------------------------------------
# 7. test_truncate_kv_cache_shape
# ---------------------------------------------------------------------------

def test_truncate_kv_cache_shape():
    n_layers, n_heads, seq_len, head_dim = 2, 2, 8, 16
    kv = _make_kv(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len, head_dim=head_dim)
    truncated = truncate_kv_cache(kv, 3)
    assert len(truncated) == n_layers
    for k, v in truncated:
        assert k.shape == (1, n_heads, 3, head_dim)
        assert v.shape == (1, n_heads, 3, head_dim)


# ---------------------------------------------------------------------------
# 8. test_merge_kv_caches_shape
# ---------------------------------------------------------------------------

def test_merge_kv_caches_shape():
    n_layers, n_heads, head_dim = 2, 2, 16
    kv_a = _make_kv(n_layers=n_layers, n_heads=n_heads, seq_len=3, head_dim=head_dim)
    kv_b = _make_kv(n_layers=n_layers, n_heads=n_heads, seq_len=5, head_dim=head_dim)
    merged = merge_kv_caches(kv_a, kv_b)
    assert len(merged) == n_layers
    for k, v in merged:
        assert k.shape == (1, n_heads, 8, head_dim)  # 3 + 5
        assert v.shape == (1, n_heads, 8, head_dim)


# ---------------------------------------------------------------------------
# 9. test_prefix_cache_put_and_get
# ---------------------------------------------------------------------------

def test_prefix_cache_put_and_get():
    cfg = PrefixCacheConfig(min_prefix_len=2)
    cache = PrefixCache(cfg)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    kv = _make_kv()
    cache.put(ids, kv)

    result_kv, match_len = cache.get(ids)
    assert result_kv is not None
    assert match_len == len(ids)
    assert len(result_kv) == len(kv)


# ---------------------------------------------------------------------------
# 10. test_prefix_cache_min_length
# ---------------------------------------------------------------------------

def test_prefix_cache_min_length():
    cfg = PrefixCacheConfig(min_prefix_len=8)
    cache = PrefixCache(cfg)

    # Too short — should not be cached
    short_ids = [1, 2, 3]
    cache.put(short_ids, _make_kv())
    assert len(cache) == 0

    # Long enough — should be cached
    long_ids = list(range(10))
    cache.put(long_ids, _make_kv())
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# 11. test_prefix_cache_max_entries_lru_eviction
# ---------------------------------------------------------------------------

def test_prefix_cache_max_entries_lru_eviction():
    cfg = PrefixCacheConfig(max_entries=2, min_prefix_len=1, eviction_policy="lru")
    cache = PrefixCache(cfg)

    ids_a = list(range(10))
    ids_b = list(range(10, 20))
    ids_c = list(range(20, 30))

    cache.put(ids_a, _make_kv())
    time.sleep(0.01)
    cache.put(ids_b, _make_kv())

    # Access ids_a to make ids_b the LRU
    cache.get(ids_a)

    # Adding ids_c should evict ids_b (least recently used)
    cache.put(ids_c, _make_kv())

    assert len(cache) == 2
    kv_b, match_b = cache.get(ids_b)
    assert kv_b is None or match_b == 0, "ids_b should have been evicted (LRU)"

    kv_a, match_a = cache.get(ids_a)
    assert kv_a is not None and match_a > 0

    kv_c, match_c = cache.get(ids_c)
    assert kv_c is not None and match_c > 0


# ---------------------------------------------------------------------------
# 12. test_prefix_cache_lfu_eviction
# ---------------------------------------------------------------------------

def test_prefix_cache_lfu_eviction():
    cfg = PrefixCacheConfig(max_entries=2, min_prefix_len=1, eviction_policy="lfu")
    cache = PrefixCache(cfg)

    ids_a = list(range(10))
    ids_b = list(range(10, 20))
    ids_c = list(range(20, 30))

    cache.put(ids_a, _make_kv())
    cache.put(ids_b, _make_kv())

    # Access ids_b multiple times so ids_a is the LFU
    cache.get(ids_b)
    cache.get(ids_b)

    # Adding ids_c should evict ids_a (least frequently used, hit_count=0)
    cache.put(ids_c, _make_kv())

    assert len(cache) == 2
    kv_a, match_a = cache.get(ids_a)
    assert kv_a is None or match_a == 0, "ids_a should have been evicted (LFU)"

    kv_b, match_b = cache.get(ids_b)
    assert kv_b is not None and match_b > 0

    kv_c, match_c = cache.get(ids_c)
    assert kv_c is not None and match_c > 0


# ---------------------------------------------------------------------------
# 13. test_prefix_cache_stats_keys
# ---------------------------------------------------------------------------

def test_prefix_cache_stats_keys():
    cfg = PrefixCacheConfig(min_prefix_len=1)
    cache = PrefixCache(cfg)
    cache.put(list(range(10)), _make_kv())
    s = cache.stats()
    assert "size" in s
    assert "hits" in s
    assert "misses" in s
    assert "hit_rate" in s


# ---------------------------------------------------------------------------
# 14. test_prefix_cache_hit_rate
# ---------------------------------------------------------------------------

def test_prefix_cache_hit_rate():
    cfg = PrefixCacheConfig(min_prefix_len=1)
    cache = PrefixCache(cfg)

    ids = list(range(10))
    cache.put(ids, _make_kv())

    # 2 hits
    cache.get(ids)
    cache.get(ids)

    # 1 miss
    cache.get(list(range(100, 110)))

    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert abs(s["hit_rate"] - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# 15. test_simulate_prefix_caching_keys
# ---------------------------------------------------------------------------

def test_simulate_prefix_caching_keys():
    cfg = PrefixCacheConfig(min_prefix_len=1)
    cache = PrefixCache(cfg)
    requests = [list(range(10)), list(range(10, 20))]
    result = simulate_prefix_caching(requests, cache, n_layers=2, n_heads=2, head_dim=16)
    assert "total_tokens" in result
    assert "reused_tokens" in result
    assert "reuse_fraction" in result
    assert "hit_rate" in result


# ---------------------------------------------------------------------------
# 16. test_simulate_prefix_caching_reuse
# ---------------------------------------------------------------------------

def test_simulate_prefix_caching_reuse():
    cfg = PrefixCacheConfig(min_prefix_len=1)
    cache = PrefixCache(cfg)

    # req1 is a strict prefix of req2; after req1 is cached, req2 will hit on it
    req1 = list(range(8))
    req2 = list(range(8)) + [100, 101, 102]

    # First request: no cache hit (cache is empty)
    # Second request: should hit on req1 (which is a prefix of req2)
    result = simulate_prefix_caching(
        [req1, req2], cache, n_layers=2, n_heads=2, head_dim=16
    )

    assert result["reuse_fraction"] > 0.0
    assert result["reused_tokens"] > 0
