"""Unit tests for src/inference/radix_cache.py (SGLang RadixAttention, 2024).

Tests cover: config defaults, prefix matching (empty / partial / full / no
match), insert, ref/deref, LRU eviction with referenced-block protection,
block counting, cache hit rate, stats(), and multiple diverging prefixes.
"""

from __future__ import annotations

import pytest

from src.inference.radix_cache import (
    CacheBlock,
    RadixCache,
    RadixCacheConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(block_id: int, token_ids: tuple[int, ...], last_access: float = 0.0) -> CacheBlock:
    b = CacheBlock(block_id=block_id, token_ids=token_ids)
    b.last_access = last_access
    return b


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RadixCacheConfig()
    assert cfg.max_blocks == 1024
    assert cfg.block_size == 16
    assert cfg.eviction_policy == "lru"


# ---------------------------------------------------------------------------
# 2. test_match_prefix_empty_cache
# ---------------------------------------------------------------------------


def test_match_prefix_empty_cache():
    cache = RadixCache()
    matched_len, block = cache.match_prefix([1, 2, 3])
    assert matched_len == 0
    assert block is None


# ---------------------------------------------------------------------------
# 3. test_insert_and_match
# ---------------------------------------------------------------------------


def test_insert_and_match():
    cache = RadixCache()
    b = _block(1, (10, 20, 30))
    cache.insert([10, 20, 30], b)

    matched_len, last_block = cache.match_prefix([10, 20, 30])
    assert matched_len == 3
    assert last_block is b


# ---------------------------------------------------------------------------
# 4. test_partial_match
# ---------------------------------------------------------------------------


def test_partial_match():
    """Insert ABCD (1,2,3,4), query ABCE (1,2,3,5) → match len 3 at most."""
    cache = RadixCache()
    # Insert prefix 1,2,3 with one block and then 1,2,3,4 with another.
    b3 = _block(1, (1, 2, 3))
    b4 = _block(2, (1, 2, 3, 4))
    cache.insert([1, 2, 3], b3)
    cache.insert([1, 2, 3, 4], b4)

    # Query 1,2,3,5 — diverges at position 4, so matched = 3 (the node for [3] has a block).
    matched_len, last_block = cache.match_prefix([1, 2, 3, 5])
    assert matched_len == 3
    assert last_block is b3


# ---------------------------------------------------------------------------
# 5. test_full_match
# ---------------------------------------------------------------------------


def test_full_match():
    cache = RadixCache()
    b = _block(42, (7, 8, 9))
    cache.insert([7, 8, 9], b)

    matched_len, last_block = cache.match_prefix([7, 8, 9])
    assert matched_len == 3
    assert last_block is b


# ---------------------------------------------------------------------------
# 6. test_no_match
# ---------------------------------------------------------------------------


def test_no_match():
    cache = RadixCache()
    b = _block(1, (100, 200))
    cache.insert([100, 200], b)

    matched_len, last_block = cache.match_prefix([999, 888])
    assert matched_len == 0
    assert last_block is None


# ---------------------------------------------------------------------------
# 7. test_ref_deref
# ---------------------------------------------------------------------------


def test_ref_deref():
    cache = RadixCache()
    b = _block(5, (1, 2))
    cache.insert([1, 2], b)

    assert b.ref_count == 0

    result = cache.ref(5)
    assert result is True
    assert b.ref_count == 1

    result = cache.ref(5)
    assert b.ref_count == 2

    result = cache.deref(5)
    assert result is True
    assert b.ref_count == 1

    cache.deref(5)
    assert b.ref_count == 0

    # Clamp at 0
    cache.deref(5)
    assert b.ref_count == 0


# ---------------------------------------------------------------------------
# 8. test_evict_lru
# ---------------------------------------------------------------------------


def test_evict_lru():
    cache = RadixCache(RadixCacheConfig(eviction_policy="lru"))

    b_old = _block(1, (1,), last_access=1.0)
    b_mid = _block(2, (2,), last_access=2.0)
    b_new = _block(3, (3,), last_access=3.0)

    cache.insert([1], b_old)
    cache.insert([2], b_mid)
    cache.insert([3], b_new)

    evicted = cache.evict(1)
    assert len(evicted) == 1
    assert evicted[0].block_id == 1  # LRU = oldest last_access


# ---------------------------------------------------------------------------
# 9. test_evict_skips_referenced
# ---------------------------------------------------------------------------


def test_evict_skips_referenced():
    cache = RadixCache(RadixCacheConfig(eviction_policy="lru"))

    b1 = _block(1, (1,), last_access=1.0)
    b2 = _block(2, (2,), last_access=2.0)

    cache.insert([1], b1)
    cache.insert([2], b2)

    # Pin b1 (oldest) so it cannot be evicted.
    cache.ref(1)

    evicted = cache.evict(1)
    assert len(evicted) == 1
    assert evicted[0].block_id == 2  # b1 is protected; b2 is next oldest

    # b1 still in cache
    assert cache.total_blocks() == 1
    assert b1.ref_count == 1


# ---------------------------------------------------------------------------
# 10. test_total_blocks
# ---------------------------------------------------------------------------


def test_total_blocks():
    cache = RadixCache()
    assert cache.total_blocks() == 0

    cache.insert([1], _block(1, (1,)))
    assert cache.total_blocks() == 1

    cache.insert([2], _block(2, (2,)))
    assert cache.total_blocks() == 2


# ---------------------------------------------------------------------------
# 11. test_cache_hit_rate_all_miss
# ---------------------------------------------------------------------------


def test_cache_hit_rate_all_miss():
    cache = RadixCache()
    rate = cache.cache_hit_rate([[1, 2], [3, 4]])
    assert rate == 0.0


# ---------------------------------------------------------------------------
# 12. test_cache_hit_rate_all_hit
# ---------------------------------------------------------------------------


def test_cache_hit_rate_all_hit():
    cache = RadixCache()
    cache.insert([1, 2], _block(1, (1, 2)))
    cache.insert([3, 4], _block(2, (3, 4)))

    rate = cache.cache_hit_rate([[1, 2, 5], [3, 4, 6]])
    assert rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 13. test_cache_hit_rate_partial
# ---------------------------------------------------------------------------


def test_cache_hit_rate_partial():
    cache = RadixCache()
    cache.insert([1, 2], _block(1, (1, 2)))

    # Query 1: hits (prefix [1,2] matches), Query 2: miss
    rate = cache.cache_hit_rate([[1, 2, 3], [99, 88]])
    assert rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 14. test_stats_keys
# ---------------------------------------------------------------------------


def test_stats_keys():
    cache = RadixCache()
    cache.insert([10], _block(1, (10,)))
    cache.cache_hit_rate([[10, 20]])

    s = cache.stats()
    assert "total_blocks" in s
    assert "free_blocks" in s
    assert "hit_rate" in s
    assert s["total_blocks"] == 1
    assert s["hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. test_multiple_prefixes
# ---------------------------------------------------------------------------


def test_multiple_prefixes():
    """Insert two diverging prefixes; both should match independently."""
    cache = RadixCache()

    # Shared prefix [1, 2], then diverge at position 3.
    b_shared = _block(0, (1, 2))
    b_left = _block(1, (1, 2, 3))
    b_right = _block(2, (1, 2, 4))

    cache.insert([1, 2], b_shared)
    cache.insert([1, 2, 3], b_left)
    cache.insert([1, 2, 4], b_right)

    # Query left branch
    ml, blk = cache.match_prefix([1, 2, 3, 99])
    assert ml == 3
    assert blk is b_left

    # Query right branch
    ml2, blk2 = cache.match_prefix([1, 2, 4, 99])
    assert ml2 == 3
    assert blk2 is b_right

    # Query with only the shared prefix part
    ml3, blk3 = cache.match_prefix([1, 2, 9])
    assert ml3 == 2
    assert blk3 is b_shared


# ---------------------------------------------------------------------------
# 16. test_ref_deref_nonexistent
# ---------------------------------------------------------------------------


def test_ref_deref_nonexistent():
    """ref/deref return False for unknown block_ids."""
    cache = RadixCache()
    assert cache.ref(9999) is False
    assert cache.deref(9999) is False
