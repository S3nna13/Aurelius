"""Tests for prefix_cache module."""

import time
import torch
import pytest

from src.inference.prefix_cache import (
    PrefixCacheConfig,
    CacheEntry,
    PrefixCache,
    PrefixCachedInference,
    hash_prefix,
    find_longest_prefix_match,
    compute_cache_hit_rate,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _dummy_kv(n_layers: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)) for _ in range(n_layers)]


def _make_cache_entry(prefix_ids: list[int], ts: float | None = None) -> CacheEntry:
    now = ts if ts is not None else time.time()
    return CacheEntry(
        prefix_hash=hash_prefix(prefix_ids),
        prefix_ids=prefix_ids,
        kv_cache=_dummy_kv(),
        n_layers=2,
        created_at=now,
        last_accessed=now,
        access_count=0,
        prefix_len=len(prefix_ids),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prefix_cache_config_defaults():
    cfg = PrefixCacheConfig()
    assert cfg.max_cached_prefixes == 64
    assert cfg.max_prefix_tokens == 512
    assert cfg.eviction_policy == "lru"
    assert cfg.compression_enabled is False
    assert cfg.ttl_seconds == 3600.0


def test_cache_entry_fields():
    now = time.time()
    entry = CacheEntry(
        prefix_hash="abc",
        prefix_ids=[1, 2, 3],
        kv_cache=_dummy_kv(),
        n_layers=2,
        created_at=now,
        last_accessed=now,
        access_count=0,
        prefix_len=3,
    )
    assert entry.prefix_hash == "abc"
    assert entry.prefix_ids == [1, 2, 3]
    assert entry.n_layers == 2
    assert entry.access_count == 0
    assert entry.prefix_len == 3


def test_hash_prefix_consistent():
    ids = [10, 20, 30]
    assert hash_prefix(ids) == hash_prefix(ids)


def test_hash_prefix_different():
    assert hash_prefix([1, 2, 3]) != hash_prefix([1, 2, 4])


def test_find_longest_prefix_match_exact():
    prefix = [1, 2, 3]
    entry = _make_cache_entry(prefix)
    cache = {entry.prefix_hash: entry}
    key, length = find_longest_prefix_match([1, 2, 3, 4, 5], cache)
    assert key == entry.prefix_hash
    assert length == 3


def test_find_longest_prefix_match_no_match():
    prefix = [10, 20, 30]
    entry = _make_cache_entry(prefix)
    cache = {entry.prefix_hash: entry}
    key, length = find_longest_prefix_match([1, 2, 3], cache)
    assert key is None
    assert length == 0


def test_prefix_cache_put_get():
    cfg = PrefixCacheConfig(max_cached_prefixes=8)
    pc = PrefixCache(cfg)
    ids = [5, 6, 7]
    kv = _dummy_kv()
    pc.put(ids, kv)
    result = pc.get(ids)
    assert result is not None
    assert result.prefix_ids == ids
    assert result.prefix_len == 3


def test_prefix_cache_eviction_lru():
    """At capacity, LRU entry should be evicted when a new entry is added."""
    cfg = PrefixCacheConfig(max_cached_prefixes=2, eviction_policy="lru")
    pc = PrefixCache(cfg)

    ids_a = [1, 2]
    ids_b = [3, 4]
    ids_c = [5, 6]

    pc.put(ids_a, _dummy_kv())
    time.sleep(0.01)
    pc.put(ids_b, _dummy_kv())

    # Access ids_a to make ids_b the LRU
    pc.get(ids_a)

    # Adding ids_c should evict ids_b (least recently accessed)
    pc.put(ids_c, _dummy_kv())

    assert pc.get(ids_b) is None, "ids_b should have been evicted (LRU)"
    assert pc.get(ids_a) is not None
    assert pc.get(ids_c) is not None


def test_prefix_cache_stats_keys():
    cfg = PrefixCacheConfig(max_cached_prefixes=4)
    pc = PrefixCache(cfg)
    pc.put([1], _dummy_kv())
    pc.put([2], _dummy_kv())
    s = pc.stats()
    assert "size" in s
    assert "total_accesses" in s
    assert "capacity" in s
    assert s["size"] == 2
    assert s["capacity"] == 4


def test_prefix_cache_evict_expired():
    """Entries with created_at in the past beyond ttl should be removed."""
    cfg = PrefixCacheConfig(ttl_seconds=1.0)
    pc = PrefixCache(cfg)

    old_ids = [99, 100]
    kv = _dummy_kv()
    pc.put(old_ids, kv)

    # Manually backdate the entry's created_at
    key = hash_prefix(old_ids)
    pc._cache[key].created_at = time.time() - 10.0  # 10 seconds ago

    removed = pc.evict_expired()
    assert removed == 1
    assert pc.get(old_ids) is None


def test_compute_cache_hit_rate_all_miss():
    """No cached entries → hit rate is 0.0."""
    cfg = PrefixCacheConfig()
    pc = PrefixCache(cfg)
    queries = [[1, 2, 3], [4, 5], [6]]
    rate = compute_cache_hit_rate(pc, queries)
    assert rate == 0.0


def test_prefix_cached_inference_cache_prefix():
    """cache_prefix stores a valid entry in the cache."""
    model = _make_model()
    cfg = PrefixCacheConfig(max_cached_prefixes=8)
    pc = PrefixCache(cfg)

    # Simple tokenizer: encode as ASCII codepoints, decode back
    def encode(text: str) -> list[int]:
        return [min(ord(c), 255) for c in text]

    def decode(ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    engine = PrefixCachedInference(model, pc, encode, decode)
    prefix_text = "Hello"
    length = engine.cache_prefix(prefix_text)

    assert length == len(encode(prefix_text))
    assert pc.stats()["size"] == 1

    # Verify the stored entry has the right prefix_ids
    entry = pc.get(encode(prefix_text))
    assert entry is not None
    assert entry.prefix_ids == encode(prefix_text)
