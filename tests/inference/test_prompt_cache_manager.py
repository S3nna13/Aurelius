"""Tests for src/inference/prompt_cache_manager.py"""

from __future__ import annotations

import time

import pytest
import torch

from src.inference.prompt_cache_manager import (
    CacheConfig,
    PrefixEntry,
    PrefixKVCache,
    PromptCacheManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kv(n_layers: int = 2, seq_len: int = 8, n_heads: int = 2, head_dim: int = 16):
    shape = (n_layers, n_heads, seq_len, head_dim)
    return torch.randn(*shape), torch.randn(*shape)


def _tokens(n: int, start: int = 1) -> list[int]:
    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# PrefixKVCache tests
# ---------------------------------------------------------------------------

class TestPrefixKVCache:
    def test_store_and_lookup_hit(self):
        cache = PrefixKVCache()
        ids = _tokens(100)
        k, v = _kv()
        cache.store(ids, k, v)
        entry = cache.lookup(ids)
        assert entry is not None
        assert entry.token_ids == ids

    def test_lookup_miss_returns_none(self):
        cache = PrefixKVCache()
        assert cache.lookup(_tokens(80)) is None

    def test_hit_count_increments(self):
        cache = PrefixKVCache()
        ids = _tokens(80)
        k, v = _kv()
        cache.store(ids, k, v)
        cache.lookup(ids)
        cache.lookup(ids)
        entry = cache.lookup(ids)
        assert entry.hit_count == 3

    def test_hash_tokens_deterministic(self):
        cache = PrefixKVCache()
        ids = _tokens(50)
        assert cache._hash_tokens(ids) == cache._hash_tokens(ids)

    def test_hash_tokens_different_sequences(self):
        cache = PrefixKVCache()
        assert cache._hash_tokens([1, 2, 3]) != cache._hash_tokens([3, 2, 1])

    def test_hit_rate_zero_on_no_lookups(self):
        cache = PrefixKVCache()
        assert cache.hit_rate() == 0.0

    def test_hit_rate_computed_correctly(self):
        cache = PrefixKVCache()
        ids = _tokens(80)
        k, v = _kv()
        cache.store(ids, k, v)
        cache.lookup(ids)
        cache.lookup(_tokens(80, start=200))
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_size_returns_entry_count(self):
        cache = PrefixKVCache()
        assert cache.size() == 0
        k, v = _kv()
        cache.store(_tokens(70), k, v)
        assert cache.size() == 1

    def test_evict_lru_removes_oldest(self):
        cache = PrefixKVCache(max_entries=2)
        k, v = _kv()
        ids_a = _tokens(70, start=1)
        ids_b = _tokens(70, start=100)
        cache.store(ids_a, k, v)
        time.sleep(0.01)
        cache.store(ids_b, k, v)
        evicted = cache.evict_lru()
        assert evicted is not None
        assert cache.size() == 1

    def test_evict_lru_on_empty_returns_none(self):
        cache = PrefixKVCache()
        assert cache.evict_lru() is None

    def test_auto_evict_at_capacity(self):
        cache = PrefixKVCache(max_entries=2)
        k, v = _kv()
        cache.store(_tokens(70, start=1), k, v)
        cache.store(_tokens(70, start=200), k, v)
        cache.store(_tokens(70, start=400), k, v)
        assert cache.size() == 2

    def test_store_updates_existing_entry(self):
        cache = PrefixKVCache()
        ids = _tokens(80)
        k1, v1 = _kv()
        k2, v2 = _kv()
        cache.store(ids, k1, v1)
        cache.store(ids, k2, v2)
        assert cache.size() == 1
        entry = cache.lookup(ids)
        assert torch.allclose(entry.k_cache, k2)

    def test_clear_resets_state(self):
        cache = PrefixKVCache()
        k, v = _kv()
        cache.store(_tokens(80), k, v)
        cache.lookup(_tokens(80))
        cache.clear()
        assert cache.size() == 0
        assert cache.hit_rate() == 0.0

    def test_stats_keys_present(self):
        cache = PrefixKVCache()
        s = cache.stats()
        for key in ("size", "max_entries", "hit_rate", "total_hits", "total_lookups"):
            assert key in s

    def test_last_used_updates_on_lookup(self):
        cache = PrefixKVCache()
        ids = _tokens(80)
        k, v = _kv()
        cache.store(ids, k, v)
        entry_before = cache._store[cache._hash_tokens(ids)]
        t_before = entry_before.last_used
        time.sleep(0.02)
        cache.lookup(ids)
        assert entry_before.last_used > t_before


# ---------------------------------------------------------------------------
# PromptCacheManager tests
# ---------------------------------------------------------------------------

class TestPromptCacheManager:
    def test_should_cache_within_bounds(self):
        cfg = CacheConfig(min_prefix_tokens=64, max_prefix_tokens=512)
        mgr = PromptCacheManager(config=cfg)
        assert mgr.should_cache(_tokens(100)) is True

    def test_should_cache_too_short(self):
        cfg = CacheConfig(min_prefix_tokens=64)
        mgr = PromptCacheManager(config=cfg)
        assert mgr.should_cache(_tokens(10)) is False

    def test_should_cache_too_long(self):
        cfg = CacheConfig(max_prefix_tokens=128)
        mgr = PromptCacheManager(config=cfg)
        assert mgr.should_cache(_tokens(200)) is False

    def test_cache_prefix_stores_when_valid(self):
        mgr = PromptCacheManager(config=CacheConfig(min_prefix_tokens=10))
        k, v = _kv()
        result = mgr.cache_prefix(_tokens(20), k, v)
        assert result is True
        assert mgr.stats()["size"] == 1

    def test_cache_prefix_skips_when_too_short(self):
        mgr = PromptCacheManager(config=CacheConfig(min_prefix_tokens=64))
        k, v = _kv()
        result = mgr.cache_prefix(_tokens(10), k, v)
        assert result is False
        assert mgr.stats()["size"] == 0

    def test_get_cached_prefix_returns_match(self):
        mgr = PromptCacheManager(config=CacheConfig(min_prefix_tokens=10))
        prefix = _tokens(20)
        full = prefix + _tokens(10, start=1000)
        k, v = _kv()
        mgr.cache_prefix(prefix, k, v)
        entry, plen = mgr.get_cached_prefix(full)
        assert entry is not None
        assert plen == 20

    def test_get_cached_prefix_miss(self):
        mgr = PromptCacheManager()
        entry, plen = mgr.get_cached_prefix(_tokens(100))
        assert entry is None
        assert plen == 0

    def test_get_cached_prefix_longest_match(self):
        mgr = PromptCacheManager(config=CacheConfig(min_prefix_tokens=5))
        short_prefix = _tokens(10)
        long_prefix = _tokens(20)
        full = long_prefix + [9999]
        k, v = _kv()
        mgr.cache_prefix(short_prefix, k, v)
        mgr.cache_prefix(long_prefix, k, v)
        entry, plen = mgr.get_cached_prefix(full)
        assert plen == 20

    def test_register_system_prompt_ignores_min_length(self):
        cfg = CacheConfig(min_prefix_tokens=100)
        mgr = PromptCacheManager(config=cfg)
        k, v = _kv()
        h = mgr.register_system_prompt(_tokens(5), k, v)
        assert isinstance(h, str)
        assert len(h) == 16
        assert mgr.stats()["size"] == 1

    def test_prune_expired_removes_old_entries(self):
        cfg = CacheConfig(min_prefix_tokens=5, ttl_s=0.01)
        mgr = PromptCacheManager(config=cfg)
        k, v = _kv()
        mgr.cache_prefix(_tokens(10), k, v)
        time.sleep(0.05)
        removed = mgr.prune_expired()
        assert removed == 1
        assert mgr.stats()["size"] == 0

    def test_prune_expired_keeps_fresh_entries(self):
        cfg = CacheConfig(min_prefix_tokens=5, ttl_s=60.0)
        mgr = PromptCacheManager(config=cfg)
        k, v = _kv()
        mgr.cache_prefix(_tokens(10), k, v)
        removed = mgr.prune_expired()
        assert removed == 0
        assert mgr.stats()["size"] == 1

    def test_stats_returns_dict(self):
        mgr = PromptCacheManager()
        s = mgr.stats()
        assert isinstance(s, dict)
        assert "size" in s

    def test_default_prefix_cache_created_when_none(self):
        mgr = PromptCacheManager(prefix_cache=None, config=None)
        assert mgr._cache is not None
        assert mgr._config is not None

    def test_expired_entries_not_returned_by_get_cached_prefix(self):
        cfg = CacheConfig(min_prefix_tokens=5, ttl_s=0.01)
        mgr = PromptCacheManager(config=cfg)
        prefix = _tokens(10)
        k, v = _kv()
        mgr.cache_prefix(prefix, k, v)
        time.sleep(0.05)
        entry, plen = mgr.get_cached_prefix(prefix + [9999])
        assert entry is None
        assert plen == 0
