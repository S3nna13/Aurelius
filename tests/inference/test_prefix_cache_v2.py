"""Tests for src/inference/prefix_cache_v2.py

Covers:
- CacheConfig defaults and validation
- CacheKey hashing and equality
- PrefixCache lifecycle (empty, store, lookup, evict, clear)
- All three eviction policies (lru, lfu, fifo)
- max_prefix_len enforcement
- hit_rate tracking
- compute_prefix_savings
- find_common_prefix
"""

from __future__ import annotations

import time

import pytest
import torch

from src.inference.prefix_cache_v2 import (
    CacheConfig,
    CacheKey,
    PrefixCache,
    compute_prefix_savings,
    find_common_prefix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_states(n_layers: int = 2, d_model: int = 8, seq_len: int = 4) -> list[torch.Tensor]:
    """Return a list of random tensors, one per layer, shape (seq_len, d_model)."""
    return [torch.randn(seq_len, d_model) for _ in range(n_layers)]


def _small_config(**kwargs) -> CacheConfig:
    """Return a CacheConfig with small defaults suitable for unit tests."""
    defaults = dict(max_entries=8, max_prefix_len=16, d_model=8, n_layers=2)
    defaults.update(kwargs)
    return CacheConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. CacheConfig defaults
# ---------------------------------------------------------------------------


def test_cache_config_defaults():
    cfg = CacheConfig()
    assert cfg.max_entries == 128
    assert cfg.max_prefix_len == 512
    assert cfg.d_model == 512
    assert cfg.n_layers == 12
    assert cfg.eviction_policy == "lru"


# ---------------------------------------------------------------------------
# 2. CacheConfig rejects invalid eviction_policy
# ---------------------------------------------------------------------------


def test_cache_config_invalid_policy():
    with pytest.raises(ValueError, match="eviction_policy"):
        CacheConfig(eviction_policy="random")


# ---------------------------------------------------------------------------
# 3. CacheKey hash and equality — same token ids
# ---------------------------------------------------------------------------


def test_cache_key_equality_same():
    k1 = CacheKey((1, 2, 3))
    k2 = CacheKey((1, 2, 3))
    assert k1 == k2
    assert hash(k1) == hash(k2)


# ---------------------------------------------------------------------------
# 4. CacheKey inequality — different token ids
# ---------------------------------------------------------------------------


def test_cache_key_inequality_different():
    k1 = CacheKey((1, 2, 3))
    k2 = CacheKey((1, 2, 4))
    assert k1 != k2


# ---------------------------------------------------------------------------
# 5. CacheKey usable as dict key
# ---------------------------------------------------------------------------


def test_cache_key_as_dict_key():
    d = {}
    k = CacheKey((10, 20))
    d[k] = "value"
    assert d[CacheKey((10, 20))] == "value"


# ---------------------------------------------------------------------------
# 6. PrefixCache is initially empty
# ---------------------------------------------------------------------------


def test_prefix_cache_initially_empty():
    cache = PrefixCache(_small_config())
    assert cache.size() == 0
    assert cache.hit_rate() == 0.0


# ---------------------------------------------------------------------------
# 7. lookup returns None on empty cache
# ---------------------------------------------------------------------------


def test_lookup_returns_none_on_empty():
    cache = PrefixCache(_small_config())
    result = cache.lookup([1, 2, 3])
    assert result is None


# ---------------------------------------------------------------------------
# 8. store adds an entry
# ---------------------------------------------------------------------------


def test_store_adds_entry():
    cache = PrefixCache(_small_config())
    cache.store([1, 2, 3], _make_kv_states())
    assert cache.size() == 1


# ---------------------------------------------------------------------------
# 9. lookup finds exact match
# ---------------------------------------------------------------------------


def test_lookup_exact_match():
    cache = PrefixCache(_small_config())
    ids = [5, 6, 7, 8]
    kv = _make_kv_states()
    cache.store(ids, kv)

    result = cache.lookup(ids)
    assert result is not None
    prefix_len, kv_out = result
    assert prefix_len == len(ids)
    assert len(kv_out) == len(kv)


# ---------------------------------------------------------------------------
# 10. lookup finds longest prefix (not full match)
# ---------------------------------------------------------------------------


def test_lookup_longest_prefix():
    cache = PrefixCache(_small_config())
    short = [1, 2, 3]
    long_ = [1, 2, 3, 4, 5]
    cache.store(short, _make_kv_states(seq_len=3))
    cache.store(long_, _make_kv_states(seq_len=5))

    # Query extends beyond both cached prefixes — should match the longer one.
    result = cache.lookup([1, 2, 3, 4, 5, 6, 7])
    assert result is not None
    prefix_len, _ = result
    assert prefix_len == 5

    # Query only shares first 3 tokens — should match *short*.
    result2 = cache.lookup([1, 2, 3, 99, 100])
    assert result2 is not None
    assert result2[0] == 3


# ---------------------------------------------------------------------------
# 11. store respects max_entries (evicts when full)
# ---------------------------------------------------------------------------


def test_store_respects_max_entries():
    cfg = _small_config(max_entries=3)
    cache = PrefixCache(cfg)

    for i in range(4):
        cache.store(list(range(i * 10, i * 10 + 4)), _make_kv_states())

    assert cache.size() == 3  # never exceeds max_entries


# ---------------------------------------------------------------------------
# 12. store ignores sequences longer than max_prefix_len
# ---------------------------------------------------------------------------


def test_store_ignores_too_long_sequences():
    cfg = _small_config(max_prefix_len=4)
    cache = PrefixCache(cfg)
    cache.store([1, 2, 3, 4, 5], _make_kv_states())  # length 5 > 4
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# 13. hit_rate is 0 initially, increases after hits
# ---------------------------------------------------------------------------


def test_hit_rate_increases_after_hit():
    cache = PrefixCache(_small_config())
    ids = [10, 20, 30]
    cache.store(ids, _make_kv_states())

    assert cache.hit_rate() == 0.0  # no lookups yet

    cache.lookup(ids)  # hit
    cache.lookup(ids)  # hit
    cache.lookup([99])  # miss

    # 2 hits out of 3 lookups
    assert abs(cache.hit_rate() - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# 14. evict removes exactly one entry
# ---------------------------------------------------------------------------


def test_evict_removes_one_entry():
    cache = PrefixCache(_small_config())
    cache.store([1, 2], _make_kv_states())
    cache.store([3, 4], _make_kv_states())
    assert cache.size() == 2

    cache.evict()
    assert cache.size() == 1


# ---------------------------------------------------------------------------
# 15. evict on empty cache is a no-op
# ---------------------------------------------------------------------------


def test_evict_on_empty_is_noop():
    cache = PrefixCache(_small_config())
    cache.evict()  # must not raise
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# 16. clear empties the cache
# ---------------------------------------------------------------------------


def test_clear_empties_cache():
    cache = PrefixCache(_small_config())
    for i in range(3):
        cache.store([i, i + 1], _make_kv_states())
    cache.lookup([0, 1])
    assert cache.size() > 0

    cache.clear()
    assert cache.size() == 0
    assert cache.hit_rate() == 0.0


# ---------------------------------------------------------------------------
# 17. LRU eviction policy
# ---------------------------------------------------------------------------


def test_lru_eviction_policy():
    cfg = _small_config(max_entries=2, eviction_policy="lru")
    cache = PrefixCache(cfg)

    ids_a = [1, 2, 3, 4]
    ids_b = [5, 6, 7, 8]
    ids_c = [9, 10, 11, 12]

    cache.store(ids_a, _make_kv_states())
    time.sleep(0.01)
    cache.store(ids_b, _make_kv_states())

    # Access ids_a — now ids_b is the least recently used
    cache.lookup(ids_a)

    # Adding ids_c must evict ids_b (LRU)
    cache.store(ids_c, _make_kv_states())
    assert cache.size() == 2
    assert cache.lookup(ids_b) is None  # evicted
    assert cache.lookup(ids_a) is not None
    assert cache.lookup(ids_c) is not None


# ---------------------------------------------------------------------------
# 18. LFU eviction policy
# ---------------------------------------------------------------------------


def test_lfu_eviction_policy():
    cfg = _small_config(max_entries=2, eviction_policy="lfu")
    cache = PrefixCache(cfg)

    ids_a = [1, 2, 3, 4]
    ids_b = [5, 6, 7, 8]
    ids_c = [9, 10, 11, 12]

    cache.store(ids_a, _make_kv_states())
    cache.store(ids_b, _make_kv_states())

    # ids_b accessed twice → ids_a is the least frequently used
    cache.lookup(ids_b)
    cache.lookup(ids_b)

    cache.store(ids_c, _make_kv_states())
    assert cache.size() == 2
    assert cache.lookup(ids_a) is None  # evicted (LFU)
    assert cache.lookup(ids_b) is not None
    assert cache.lookup(ids_c) is not None


# ---------------------------------------------------------------------------
# 19. FIFO eviction policy
# ---------------------------------------------------------------------------


def test_fifo_eviction_policy():
    cfg = _small_config(max_entries=2, eviction_policy="fifo")
    cache = PrefixCache(cfg)

    ids_a = [1, 2, 3, 4]
    ids_b = [5, 6, 7, 8]
    ids_c = [9, 10, 11, 12]

    cache.store(ids_a, _make_kv_states())
    cache.store(ids_b, _make_kv_states())

    # Access ids_a many times — but FIFO should still evict ids_a (first in)
    cache.lookup(ids_a)
    cache.lookup(ids_a)

    cache.store(ids_c, _make_kv_states())
    assert cache.size() == 2
    assert cache.lookup(ids_a) is None  # evicted (FIFO, inserted first)
    assert cache.lookup(ids_b) is not None
    assert cache.lookup(ids_c) is not None


# ---------------------------------------------------------------------------
# 20. compute_prefix_savings in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_prefix_savings_range():
    savings = compute_prefix_savings(original_len=100, cached_prefix_len=40, n_layers=6)
    assert 0.0 <= savings <= 1.0
    assert abs(savings - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# 21. compute_prefix_savings = 0 when no prefix cached
# ---------------------------------------------------------------------------


def test_compute_prefix_savings_no_prefix():
    savings = compute_prefix_savings(original_len=50, cached_prefix_len=0, n_layers=6)
    assert savings == 0.0


# ---------------------------------------------------------------------------
# 22. compute_prefix_savings = 0 for zero original_len
# ---------------------------------------------------------------------------


def test_compute_prefix_savings_zero_original():
    savings = compute_prefix_savings(original_len=0, cached_prefix_len=0, n_layers=6)
    assert savings == 0.0


# ---------------------------------------------------------------------------
# 23. compute_prefix_savings = 1.0 for full prefix
# ---------------------------------------------------------------------------


def test_compute_prefix_savings_full():
    savings = compute_prefix_savings(original_len=50, cached_prefix_len=50, n_layers=6)
    assert abs(savings - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 24. find_common_prefix — identical sequences
# ---------------------------------------------------------------------------


def test_find_common_prefix_identical():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert find_common_prefix(seqs) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 25. find_common_prefix — partial overlap
# ---------------------------------------------------------------------------


def test_find_common_prefix_partial():
    seqs = [[1, 2, 3, 99], [1, 2, 3, 100], [1, 2, 3, 200]]
    assert find_common_prefix(seqs) == [1, 2, 3]


# ---------------------------------------------------------------------------
# 26. find_common_prefix — empty for no common prefix
# ---------------------------------------------------------------------------


def test_find_common_prefix_no_common():
    seqs = [[1, 2, 3], [4, 5, 6]]
    assert find_common_prefix(seqs) == []


# ---------------------------------------------------------------------------
# 27. find_common_prefix — empty input
# ---------------------------------------------------------------------------


def test_find_common_prefix_empty_input():
    assert find_common_prefix([]) == []


# ---------------------------------------------------------------------------
# 28. find_common_prefix — single sequence
# ---------------------------------------------------------------------------


def test_find_common_prefix_single_sequence():
    seqs = [[7, 8, 9]]
    assert find_common_prefix(seqs) == [7, 8, 9]


# ---------------------------------------------------------------------------
# 29. lookup increments hit_count on entry
# ---------------------------------------------------------------------------


def test_lookup_increments_hit_count():
    cache = PrefixCache(_small_config())
    ids = [1, 2, 3]
    cache.store(ids, _make_kv_states())

    # Peek at the entry via the internal store
    key = CacheKey(tuple(ids))
    entry = cache._store[key]
    assert entry.hit_count == 0

    cache.lookup(ids)
    assert entry.hit_count == 1
    cache.lookup(ids)
    assert entry.hit_count == 2


# ---------------------------------------------------------------------------
# 30. store with sequence exactly equal to max_prefix_len is accepted
# ---------------------------------------------------------------------------


def test_store_at_max_prefix_len_boundary():
    cfg = _small_config(max_prefix_len=5)
    cache = PrefixCache(cfg)
    cache.store([1, 2, 3, 4, 5], _make_kv_states())  # exactly 5 — allowed
    assert cache.size() == 1

    cache.store([1, 2, 3, 4, 5, 6], _make_kv_states())  # 6 > 5 — rejected
    assert cache.size() == 1
