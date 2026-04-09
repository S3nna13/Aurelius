"""Tests for src/inference/kv_cache_manager.py"""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.kv_cache_manager import (
    KVCacheConfig,
    KVCacheManager,
    LRUCache,
    ScoreBasedCache,
    chunked_prefill,
    compute_cache_memory_mb,
    compute_max_seq_from_budget,
)


# ---------------------------------------------------------------------------
# KVCacheConfig
# ---------------------------------------------------------------------------


def test_kvcacheconfig_defaults():
    cfg = KVCacheConfig()
    assert cfg.max_seq_len == 2048
    assert cfg.eviction_policy == "lru"
    assert cfg.memory_budget_mb == 512.0
    assert cfg.prefill_chunk_size == 512
    assert cfg.n_layers == 24
    assert cfg.n_heads == 8
    assert cfg.head_dim == 64


# ---------------------------------------------------------------------------
# compute_cache_memory_mb
# ---------------------------------------------------------------------------


def test_compute_cache_memory_mb_formula():
    n_layers, n_heads, head_dim, seq_len, dtype_bytes = 2, 4, 32, 128, 2
    expected = 2 * n_layers * n_heads * seq_len * head_dim * dtype_bytes / (1024 ** 2)
    result = compute_cache_memory_mb(n_layers, n_heads, head_dim, seq_len, dtype_bytes)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_compute_cache_memory_mb_scales_with_seq_len():
    result_1 = compute_cache_memory_mb(4, 4, 64, 512)
    result_2 = compute_cache_memory_mb(4, 4, 64, 1024)
    assert math.isclose(result_2, result_1 * 2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_max_seq_from_budget
# ---------------------------------------------------------------------------


def test_compute_max_seq_from_budget_inverse():
    n_layers, n_heads, head_dim = 4, 8, 64
    seq_len = 512
    budget = compute_cache_memory_mb(n_layers, n_heads, head_dim, seq_len)
    recovered = compute_max_seq_from_budget(budget, n_layers, n_heads, head_dim)
    assert recovered == seq_len


def test_compute_max_seq_from_budget_larger_budget():
    small = compute_max_seq_from_budget(256.0, 4, 8, 64)
    large = compute_max_seq_from_budget(512.0, 4, 8, 64)
    assert large == 2 * small


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_basic_insert():
    cache = LRUCache(capacity=3)
    assert cache.insert(1) is None
    assert cache.insert(2) is None
    assert 1 in cache
    assert 2 in cache
    assert len(cache) == 2


def test_lru_evicts_lru_when_full():
    cache = LRUCache(capacity=2)
    cache.insert(10)
    cache.insert(20)
    evicted = cache.insert(30)   # should evict 10 (LRU)
    assert evicted == 10
    assert 10 not in cache
    assert 20 in cache
    assert 30 in cache


def test_lru_access_promotes_to_mru():
    cache = LRUCache(capacity=2)
    cache.insert(1)
    cache.insert(2)
    # Access key 1 so it becomes MRU; key 2 should be evicted next
    cache.access(1)
    evicted = cache.insert(3)
    assert evicted == 2
    assert 1 in cache
    assert 3 in cache


def test_lru_evict_lru_returns_key():
    cache = LRUCache(capacity=3)
    cache.insert(5)
    cache.insert(6)
    key = cache.evict_lru()
    assert key == 5
    assert 5 not in cache


def test_lru_len():
    cache = LRUCache(capacity=5)
    for k in range(4):
        cache.insert(k)
    assert len(cache) == 4


# ---------------------------------------------------------------------------
# ScoreBasedCache
# ---------------------------------------------------------------------------


def test_score_based_evicts_lowest_score():
    cache = ScoreBasedCache(capacity=3)
    cache.insert(1, score=0.9)
    cache.insert(2, score=0.1)   # lowest
    cache.insert(3, score=0.5)
    evicted = cache.evict_lowest()
    assert evicted == 2
    assert len(cache) == 2


def test_score_based_insert_returns_evicted_when_full():
    cache = ScoreBasedCache(capacity=2)
    cache.insert(10, score=1.0)
    cache.insert(20, score=0.2)   # lower score
    evicted = cache.insert(30, score=0.8)   # evicts 20
    assert evicted == 20
    assert 20 not in cache._scores
    assert 10 in cache._scores
    assert 30 in cache._scores


def test_score_based_update_score():
    cache = ScoreBasedCache(capacity=2)
    cache.insert(1, score=0.5)
    cache.update_score(1, 0.99)
    cache.insert(2, score=0.1)
    evicted = cache.evict_lowest()
    assert evicted == 2   # key 1 now has score 0.99 > 0.1


# ---------------------------------------------------------------------------
# chunked_prefill
# ---------------------------------------------------------------------------


def test_chunked_prefill_correct_number_of_chunks():
    T = 1024
    chunk_size = 256
    prompt = torch.zeros(1, T, dtype=torch.long)
    chunks = chunked_prefill(prompt, chunk_size)
    assert len(chunks) == T // chunk_size


def test_chunked_prefill_last_chunk_shorter():
    T = 1000
    chunk_size = 256
    prompt = torch.zeros(1, T, dtype=torch.long)
    chunks = chunked_prefill(prompt, chunk_size)
    # Last chunk should have T % chunk_size = 232 tokens
    assert chunks[-1].size(1) == T % chunk_size
    # All intermediate chunks should be full-sized
    for c in chunks[:-1]:
        assert c.size(1) == chunk_size


def test_chunked_prefill_concatenates_to_original():
    T = 300
    chunk_size = 100
    prompt = torch.arange(T, dtype=torch.long).unsqueeze(0)
    chunks = chunked_prefill(prompt, chunk_size)
    reconstructed = torch.cat(chunks, dim=1)
    assert torch.equal(reconstructed, prompt)


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------


def test_kvcachemanager_allocate_and_memory_used():
    cfg = KVCacheConfig(
        n_layers=2, n_heads=4, head_dim=32, memory_budget_mb=1024.0
    )
    mgr = KVCacheManager(cfg)
    assert mgr.memory_used_mb() == 0.0

    result = mgr.allocate(seq_id=0, seq_len=128)
    assert result is True
    expected = compute_cache_memory_mb(2, 4, 32, 128)
    assert math.isclose(mgr.memory_used_mb(), expected, rel_tol=1e-9)


def test_kvcachemanager_stats_keys_present():
    cfg = KVCacheConfig(n_layers=2, n_heads=4, head_dim=32, memory_budget_mb=512.0)
    mgr = KVCacheManager(cfg)
    mgr.allocate(seq_id=1, seq_len=64)
    mgr.allocate(seq_id=2, seq_len=64)

    s = mgr.stats()
    assert "n_sequences" in s
    assert "memory_used_mb" in s
    assert "memory_budget_mb" in s
    assert "utilization" in s
    assert s["n_sequences"] == 2
    assert s["memory_budget_mb"] == 512.0
    assert 0.0 < s["utilization"] <= 1.0


def test_kvcachemanager_rejects_over_budget():
    # Use a tiny budget that cannot fit a large sequence
    cfg = KVCacheConfig(
        n_layers=24, n_heads=8, head_dim=64,
        memory_budget_mb=0.001,   # essentially 0
    )
    mgr = KVCacheManager(cfg)
    result = mgr.allocate(seq_id=0, seq_len=2048)
    assert result is False


def test_kvcachemanager_free_releases_memory():
    cfg = KVCacheConfig(n_layers=2, n_heads=4, head_dim=32, memory_budget_mb=512.0)
    mgr = KVCacheManager(cfg)
    mgr.allocate(seq_id=5, seq_len=256)
    used_before = mgr.memory_used_mb()
    mgr.free(seq_id=5)
    assert mgr.memory_used_mb() == 0.0
    assert mgr.stats()["n_sequences"] == 0
