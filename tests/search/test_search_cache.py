"""Tests for src/search/search_cache.py  (≥28 tests)."""

import time
import unittest.mock as mock

from src.search.search_cache import SEARCH_CACHE_REGISTRY, CacheEntry, SearchCache

# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_exists(self):
        assert SEARCH_CACHE_REGISTRY is not None

    def test_registry_default_key(self):
        assert "default" in SEARCH_CACHE_REGISTRY

    def test_registry_default_is_search_cache_class(self):
        assert SEARCH_CACHE_REGISTRY["default"] is SearchCache


# ---------------------------------------------------------------------------
# CacheEntry dataclass
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_cache_entry_creation(self):
        entry = CacheEntry(query="q", results=[1, 2], timestamp=0.0)
        assert entry.query == "q"
        assert entry.results == [1, 2]
        assert entry.hit_count == 0

    def test_cache_entry_hit_count_mutable(self):
        entry = CacheEntry(query="q", results=[], timestamp=0.0)
        entry.hit_count += 1
        assert entry.hit_count == 1


# ---------------------------------------------------------------------------
# Empty cache
# ---------------------------------------------------------------------------


class TestEmptyCache:
    def test_get_returns_none_on_empty(self):
        cache = SearchCache()
        assert cache.get("foo") is None

    def test_stats_empty(self):
        cache = SearchCache()
        s = cache.stats()
        assert s["size"] == 0
        assert s["hit_count"] == 0

    def test_invalidate_missing_returns_false(self):
        cache = SearchCache()
        assert cache.invalidate("ghost") is False

    def test_clear_on_empty_is_noop(self):
        cache = SearchCache()
        cache.clear()  # should not raise
        assert cache.stats()["size"] == 0


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------


class TestPutGet:
    def test_put_and_get(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("cats", [1, 2, 3])
        assert cache.get("cats") == [1, 2, 3]

    def test_put_stores_copy(self):
        cache = SearchCache(ttl_seconds=9999)
        results = [1, 2]
        cache.put("q", results)
        results.append(99)
        assert cache.get("q") == [1, 2]  # original copy unchanged

    def test_get_miss_returns_none(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("real", [1])
        assert cache.get("missing") is None

    def test_stats_size_after_puts(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("a", [])
        cache.put("b", [])
        assert cache.stats()["size"] == 2

    def test_stats_max_size(self):
        cache = SearchCache(max_size=10)
        assert cache.stats()["max_size"] == 10


# ---------------------------------------------------------------------------
# Hit count
# ---------------------------------------------------------------------------


class TestHitCount:
    def test_hit_count_increments_on_get(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("q", [1])
        cache.get("q")
        cache.get("q")
        assert cache.stats()["hit_count"] == 2

    def test_hit_count_zero_if_never_accessed(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("q", [1])
        assert cache.stats()["hit_count"] == 0

    def test_hit_count_sums_across_entries(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.get("a")
        cache.get("b")
        cache.get("b")
        assert cache.stats()["hit_count"] == 3


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestTTL:
    def test_expired_entry_returns_none(self):
        cache = SearchCache(ttl_seconds=60)
        cache.put("q", [42])
        # Patch time.monotonic to simulate expiry
        with mock.patch(
            "src.search.search_cache.time.monotonic", return_value=time.monotonic() + 61
        ):
            assert cache.get("q") is None

    def test_not_yet_expired_returns_result(self):
        cache = SearchCache(ttl_seconds=300)
        cache.put("q", [7])
        with mock.patch(
            "src.search.search_cache.time.monotonic", return_value=time.monotonic() + 100
        ):
            assert cache.get("q") == [7]

    def test_expired_entry_removed_from_size(self):
        cache = SearchCache(ttl_seconds=1)
        cache.put("q", [1])
        future = time.monotonic() + 5
        with mock.patch("src.search.search_cache.time.monotonic", return_value=future):
            cache.get("q")  # triggers removal
        assert cache.stats()["size"] == 0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_lru_evicts_oldest_when_full(self):
        cache = SearchCache(max_size=3, ttl_seconds=9999)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.put("c", [3])
        cache.put("d", [4])  # "a" should be evicted
        assert cache.get("a") is None
        assert cache.get("b") == [2]
        assert cache.get("c") == [3]
        assert cache.get("d") == [4]

    def test_lru_access_refreshes_order(self):
        cache = SearchCache(max_size=3, ttl_seconds=9999)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.put("c", [3])
        cache.get("a")  # "a" is now most-recently used; "b" becomes oldest
        cache.put("d", [4])  # "b" should be evicted
        assert cache.get("b") is None
        assert cache.get("a") == [1]

    def test_size_stays_at_max(self):
        cache = SearchCache(max_size=2, ttl_seconds=9999)
        for i in range(10):
            cache.put(str(i), [i])
        assert cache.stats()["size"] == 2


# ---------------------------------------------------------------------------
# Invalidate / clear
# ---------------------------------------------------------------------------


class TestInvalidateClear:
    def test_invalidate_existing_returns_true(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("q", [1])
        assert cache.invalidate("q") is True

    def test_invalidate_removes_entry(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("q", [1])
        cache.invalidate("q")
        assert cache.get("q") is None

    def test_invalidate_missing_returns_false(self):
        cache = SearchCache(ttl_seconds=9999)
        assert cache.invalidate("no_such_key") is False

    def test_clear_empties_cache(self):
        cache = SearchCache(ttl_seconds=9999)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get("a") is None
