from __future__ import annotations

import time

from src.cache import CacheService, LRUCache, SemanticCache


def test_semantic_cache_hit() -> None:
    cache = SemanticCache(threshold=0.5)
    cache.set("hello world", "result_a")
    result = cache.get("hello")
    assert result is not None


def test_semantic_cache_miss() -> None:
    cache = SemanticCache(threshold=0.99)
    cache.set("hello world", "result_a")
    result = cache.get("completely different query")
    assert result is None


def test_semantic_cache_clear() -> None:
    cache = SemanticCache()
    cache.set("a", 1)
    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0


def test_lru_cache_basic() -> None:
    cache = LRUCache(capacity=3, default_ttl=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None


def test_lru_cache_eviction() -> None:
    cache = LRUCache(capacity=2, default_ttl=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_ttl() -> None:
    cache = LRUCache(capacity=10, default_ttl=0.001)
    cache.set("a", 1)
    time.sleep(0.005)
    assert cache.get("a") is None


def test_cache_service_combined() -> None:
    service = CacheService()
    service.set("what is python", "A programming language", key="python_fact")
    result = service.get("tell me about python", key="python_fact")
    assert result is not None


def test_cache_service_stats() -> None:
    service = CacheService()
    service.set("a", 1, key="k1")
    service.set("b", 2, key="k2")
    stats = service.stats()
    assert stats["semantic_entries"] == 2
    assert stats["lru_entries"] == 2
