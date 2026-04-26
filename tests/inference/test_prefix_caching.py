"""Tests for src/inference/prefix_caching.py"""

from __future__ import annotations

import time

import pytest

from src.inference.prefix_caching import (
    CachedPrefix,
    PrefixCache,
    PrefixCacheConfig,
    compute_prefix_hash,
)


class TestPrefixCacheConfig:
    def test_default_max_size(self):
        cfg = PrefixCacheConfig()
        assert cfg.max_size == 64

    def test_default_min_prefix_len(self):
        cfg = PrefixCacheConfig()
        assert cfg.min_prefix_len == 4

    def test_custom_max_size(self):
        cfg = PrefixCacheConfig(max_size=128)
        assert cfg.max_size == 128

    def test_custom_min_prefix_len(self):
        cfg = PrefixCacheConfig(min_prefix_len=1)
        assert cfg.min_prefix_len == 1


class TestComputePrefixHash:
    def test_deterministic(self):
        ids = [1, 2, 3, 4]
        assert compute_prefix_hash(ids) == compute_prefix_hash(ids)

    def test_different_inputs_differ(self):
        assert compute_prefix_hash([1, 2, 3]) != compute_prefix_hash([1, 2, 4])

    def test_order_matters(self):
        assert compute_prefix_hash([1, 2, 3]) != compute_prefix_hash([3, 2, 1])

    def test_empty_list(self):
        h = compute_prefix_hash([])
        assert isinstance(h, str) and len(h) > 0


class TestPrefixCacheConstruction:
    def test_default_config(self):
        cache = PrefixCache()
        assert cache.config.max_size == 64
        assert cache.config.min_prefix_len == 4

    def test_custom_config(self):
        cfg = PrefixCacheConfig(max_size=10, min_prefix_len=1)
        cache = PrefixCache(config=cfg)
        assert cache.config.max_size == 10
        assert cache.config.min_prefix_len == 1

    def test_empty_cache_length(self):
        cache = PrefixCache()
        assert len(cache) == 0


class TestPrefixCachePutGet:
    def test_put_and_get_exact(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        ids = [10, 20, 30, 40, 50]
        kv = {"dummy": "state"}
        cache.put(ids, kv)
        result, match_len = cache.get(ids)
        assert result is not None
        assert result == kv
        assert match_len == len(ids)

    def test_get_miss_returns_none(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        result, match_len = cache.get([99, 98, 97])
        assert result is None
        assert match_len == 0

    def test_put_below_min_prefix_not_cached(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=8))
        cache.put([1, 2, 3], {"kv": "data"})
        assert len(cache) == 0

    def test_put_above_min_prefix_is_cached(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=2))
        cache.put([1, 2, 3, 4, 5], {"kv": "data"})
        assert len(cache) == 1

    def test_update_existing_key(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        ids = [1, 2, 3]
        cache.put(ids, {"v1": "old"})
        cache.put(ids, {"v1": "new"})
        assert len(cache) == 1
        result, _ = cache.get(ids)
        assert result == {"v1": "new"}

    def test_put_updates_last_accessed(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        ids = [1, 2, 3]
        cache.put(ids, {"x": 1})
        entry = cache._cache[compute_prefix_hash(ids)]
        original_ts = entry.last_accessed
        time.sleep(0.01)
        cache.put(ids, {"x": 2})
        assert entry.last_accessed > original_ts


class TestPrefixCacheLRUEviction:
    def test_evicts_oldest_when_full(self):
        cfg = PrefixCacheConfig(max_size=2, min_prefix_len=1)
        cache = PrefixCache(cfg)
        cache.put([1, 2, 3], "a")
        cache.put([4, 5, 6], "b")
        cache.put([7, 8, 9], "c")
        assert len(cache) == 2
        result_a, _ = cache.get([1, 2, 3])
        assert result_a is None

    def test_get_refreshes_lru_position(self):
        cfg = PrefixCacheConfig(max_size=2, min_prefix_len=1)
        cache = PrefixCache(cfg)
        cache.put([1, 2, 3], "a")
        cache.put([4, 5, 6], "b")
        cache.get([1, 2, 3])
        cache.put([7, 8, 9], "c")
        assert len(cache) == 2
        result_b, _ = cache.get([4, 5, 6])
        assert result_b is None
        result_a, _ = cache.get([1, 2, 3])
        assert result_a == "a"


class TestPrefixCacheStats:
    def test_stats_keys(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        s = cache.stats()
        assert "size" in s
        assert "max_size" in s
        assert "hits" in s
        assert "misses" in s
        assert "hit_rate" in s

    def test_stats_hit_rate(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        ids = [1, 2, 3, 4]
        cache.put(ids, "v")
        cache.get(ids)
        cache.get(ids)
        cache.get([9, 8, 7])
        s = cache.stats()
        assert s["hits"] == 2.0
        assert s["misses"] == 1.0
        assert abs(s["hit_rate"] - 2.0 / 3.0) < 1e-9

    def test_stats_empty_cache(self):
        cache = PrefixCache()
        s = cache.stats()
        assert s["hit_rate"] == 0.0
        assert s["size"] == 0.0

    def test_clear_resets_stats(self):
        cache = PrefixCache(PrefixCacheConfig(min_prefix_len=1))
        cache.put([1, 2, 3], "v")
        cache.get([1, 2, 3])
        cache.clear()
        s = cache.stats()
        assert s["hits"] == 0.0
        assert s["misses"] == 0.0
        assert s["size"] == 0.0


class TestCachedPrefix:
    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(CachedPrefix)

    def test_fields(self):
        cp = CachedPrefix(prefix_ids=(1, 2, 3), kv_state={"k": "v"})
        assert cp.prefix_ids == (1, 2, 3)
        assert cp.kv_state == {"k": "v"}
        assert cp.last_accessed > 0
        assert cp.created_at > 0
