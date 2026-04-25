"""Tests for src/serving/response_cache.py"""

import time
import pytest
from src.serving.response_cache import (
    CachePolicy,
    CachedResponse,
    ResponseCache,
    SERVING_REGISTRY,
)


# ---------------------------------------------------------------------------
# CachedResponse expiry
# ---------------------------------------------------------------------------


def test_cached_response_not_expired_fresh():
    now = time.time()
    cr = CachedResponse(key="k", response="r", created_at=now, ttl=300.0)
    assert not cr.is_expired(now + 1)


def test_cached_response_expired():
    past = time.time() - 400
    cr = CachedResponse(key="k", response="r", created_at=past, ttl=300.0)
    assert cr.is_expired()


# ---------------------------------------------------------------------------
# Basic get / put
# ---------------------------------------------------------------------------


def test_put_and_get():
    c = ResponseCache()
    c.put("key1", "hello")
    assert c.get("key1") == "hello"


def test_get_missing_returns_none():
    c = ResponseCache()
    assert c.get("no-such-key") is None


def test_get_expired_returns_none():
    c = ResponseCache()
    c.put("k", "val", ttl=0.001)
    time.sleep(0.01)
    assert c.get("k") is None


def test_put_updates_existing_key():
    c = ResponseCache()
    c.put("k", "v1")
    c.put("k", "v2")
    assert c.get("k") == "v2"


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_invalidate_removes_entry():
    c = ResponseCache()
    c.put("k", "v")
    c.invalidate("k")
    assert c.get("k") is None


def test_invalidate_missing_is_noop():
    c = ResponseCache()
    c.invalidate("ghost")  # should not raise


# ---------------------------------------------------------------------------
# TTL eviction
# ---------------------------------------------------------------------------


def test_evict_expired_returns_count():
    c = ResponseCache()
    c.put("a", "va", ttl=0.001)
    c.put("b", "vb", ttl=0.001)
    c.put("c", "vc", ttl=9999.0)
    time.sleep(0.02)
    count = c.evict_expired()
    assert count == 2
    assert c.get("c") == "vc"


# ---------------------------------------------------------------------------
# LRU eviction at capacity
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_when_full():
    c = ResponseCache(capacity=3)
    c.put("a", "va")
    c.put("b", "vb")
    c.put("c", "vc")
    # Touch "a" to make it recently used
    c.get("a")
    # Adding "d" should evict LRU ("b")
    c.put("d", "vd")
    assert c.get("b") is None
    assert c.get("a") == "va"
    assert c.get("c") == "vc"
    assert c.get("d") == "vd"


# ---------------------------------------------------------------------------
# make_key
# ---------------------------------------------------------------------------


def test_make_key_deterministic():
    k1 = ResponseCache.make_key("hello", "gpt-2")
    k2 = ResponseCache.make_key("hello", "gpt-2")
    assert k1 == k2


def test_make_key_different_for_different_inputs():
    k1 = ResponseCache.make_key("hello", "model-a")
    k2 = ResponseCache.make_key("hello", "model-b")
    assert k1 != k2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_serving_registry_contains_response_cache():
    assert "response_cache" in SERVING_REGISTRY
    assert isinstance(SERVING_REGISTRY["response_cache"], ResponseCache)


def test_cache_len():
    c = ResponseCache()
    assert len(c) == 0
    c.put("x", "y")
    assert len(c) == 1
