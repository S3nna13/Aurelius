"""Unit tests for ``src.serving.prompt_cache``.

These tests assume single-threaded access; concurrent safety is not
required by the spec.
"""

from __future__ import annotations

import time

import pytest

from src.serving.prompt_cache import CachedResponse, PromptCache


def test_put_then_get_returns_cached_response():
    cache = PromptCache()
    cache.put("hello", "world")
    entry = cache.get("hello")
    assert entry is not None
    assert isinstance(entry, CachedResponse)
    assert entry.completion == "world"


def test_get_on_unknown_prompt_returns_none():
    cache = PromptCache()
    assert cache.get("never-seen") is None


def test_different_params_produce_different_entries():
    cache = PromptCache()
    cache.put("p", "a", params={"temperature": 0.1})
    cache.put("p", "b", params={"temperature": 0.9})
    assert cache.get("p", {"temperature": 0.1}).completion == "a"
    assert cache.get("p", {"temperature": 0.9}).completion == "b"


def test_ttl_expiry():
    cache = PromptCache()
    cache.put("p", "c", ttl=0.05)
    assert cache.get("p") is not None
    time.sleep(0.08)
    assert cache.get("p") is None


def test_lru_eviction_at_max_entries():
    cache = PromptCache(max_entries=2)
    cache.put("a", "1")
    cache.put("b", "2")
    # touch a so b becomes LRU
    cache.get("a")
    cache.put("c", "3")
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.stats()["evictions"] == 1


def test_stats_hits_and_misses_correct():
    cache = PromptCache()
    cache.put("p", "x")
    cache.get("p")  # hit
    cache.get("p")  # hit
    cache.get("q")  # miss
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["size"] == 1


def test_invalidate_removes_entry():
    cache = PromptCache()
    cache.put("p", "x")
    assert cache.invalidate("p") is True
    assert cache.get("p") is None
    assert cache.invalidate("p") is False


def test_clear_returns_count_and_empties():
    cache = PromptCache()
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("c", "3")
    n = cache.clear()
    assert n == 3
    assert len(cache) == 0
    assert cache.get("a") is None


def test_prune_expired_returns_count():
    cache = PromptCache()
    cache.put("a", "1", ttl=0.05)
    cache.put("b", "2", ttl=100.0)
    cache.put("c", "3", ttl=0.05)
    time.sleep(0.08)
    n = cache.prune_expired()
    assert n == 2
    assert cache.get("b") is not None
    assert len(cache) == 1


def test_custom_hasher_is_used():
    calls = []

    def fake_hasher(s: str) -> str:
        calls.append(s)
        return "fixed-key"

    cache = PromptCache(hasher=fake_hasher)
    cache.put("p", "x")
    cache.put("other", "y")  # same key → overwrites
    assert cache.get("p").completion == "y"
    assert len(calls) > 0


def test_hashing_is_deterministic():
    c1 = PromptCache()
    c2 = PromptCache()
    c1.put("hello", "a", params={"k": 1, "j": 2})
    c2.put("hello", "b", params={"j": 2, "k": 1})
    # Same prompt + same (sorted) params -> same key across caches.
    assert (
        c1.get("hello", {"k": 1, "j": 2}).prompt_hash
        == c2.get("hello", {"j": 2, "k": 1}).prompt_hash
    )


def test_hit_count_increments_on_each_get():
    cache = PromptCache()
    cache.put("p", "x")
    cache.get("p")
    cache.get("p")
    cache.get("p")
    assert cache.get("p").hit_count == 4


def test_parameter_order_independence():
    cache = PromptCache()
    cache.put("p", "x", params={"a": 1, "b": 2, "c": 3})
    assert cache.get("p", {"c": 3, "a": 1, "b": 2}) is not None
    assert cache.get("p", {"b": 2, "c": 3, "a": 1}).completion == "x"


def test_none_vs_empty_params_collide():
    # Spec-adjacent sanity: ``None`` params and ``{}`` params key the
    # same entry so callers don't accidentally bypass the cache.
    cache = PromptCache()
    cache.put("p", "x", params=None)
    assert cache.get("p", {}) is not None


def test_max_entries_must_be_positive():
    with pytest.raises(ValueError):
        PromptCache(max_entries=0)
