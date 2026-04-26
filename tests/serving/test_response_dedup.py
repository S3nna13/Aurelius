import time

from src.serving.response_dedup import (
    RESPONSE_DEDUP_REGISTRY,
    DedupStrategy,
    ResponseDedup,
)


def test_store_and_lookup_hit():
    rd = ResponseDedup()
    rd.store("hello", "world")
    assert rd.lookup("hello") == "world"


def test_lookup_miss_returns_none():
    rd = ResponseDedup()
    assert rd.lookup("missing") is None


def test_ttl_expiry():
    rd = ResponseDedup(ttl_s=0.01)
    rd.store("prompt", "response")
    time.sleep(0.05)
    assert rd.lookup("prompt") is None


def test_max_entries_eviction():
    rd = ResponseDedup(max_entries=3)
    rd.store("a", "ra")
    time.sleep(0.001)
    rd.store("b", "rb")
    time.sleep(0.001)
    rd.store("c", "rc")
    time.sleep(0.001)
    rd.store("d", "rd")
    assert rd.stats()["size"] == 3
    assert rd.lookup("d") == "rd"


def test_invalidate():
    rd = ResponseDedup()
    rd.store("p", "r")
    rd.invalidate("p")
    assert rd.lookup("p") is None


def test_stats_hit_count():
    rd = ResponseDedup()
    rd.store("q", "ans")
    rd.lookup("q")
    rd.lookup("q")
    s = rd.stats()
    assert s["hits"] == 2
    assert s["size"] == 1
    assert s["strategy"] == "exact_hash"


def test_hit_count_on_entry():
    rd = ResponseDedup()
    rd.store("q", "ans")
    rd.lookup("q")
    rd.lookup("q")
    key = rd._compute_key("q")
    assert rd._store[key].hit_count == 2


def test_clear():
    rd = ResponseDedup()
    rd.store("x", "y")
    rd.lookup("x")
    rd.clear()
    assert rd.stats()["size"] == 0
    assert rd.stats()["hits"] == 0


def test_exact_hash_strategy_key():
    rd = ResponseDedup(strategy=DedupStrategy.EXACT_HASH)
    k1 = rd._compute_key("hello")
    k2 = rd._compute_key("hello world")
    assert k1 != k2


def test_prefix_hash_strategy_key():
    rd = ResponseDedup(strategy=DedupStrategy.PREFIX_HASH)
    long_prompt = "x" * 300
    long_prompt2 = "x" * 300 + "different"
    k1 = rd._compute_key(long_prompt)
    k2 = rd._compute_key(long_prompt2)
    assert k1 == k2
    k3 = rd._compute_key("short")
    assert k1 != k3


def test_semantic_hash_normalizes_case_and_whitespace():
    rd = ResponseDedup(strategy=DedupStrategy.SEMANTIC_HASH)
    k1 = rd._compute_key("Hello   World")
    k2 = rd._compute_key("hello world")
    assert k1 == k2


def test_strategies_produce_different_keys_for_different_input():
    for strat in DedupStrategy:
        rd = ResponseDedup(strategy=strat)
        k1 = rd._compute_key("prompt one")
        k2 = rd._compute_key("prompt two")
        assert k1 != k2, f"Strategy {strat} should produce different keys"


def test_registry_key():
    assert "default" in RESPONSE_DEDUP_REGISTRY
    assert RESPONSE_DEDUP_REGISTRY["default"] is ResponseDedup
