"""Unit tests for ``src.longcontext.prefix_cache``."""

from __future__ import annotations

import time

import pytest

from src.longcontext.prefix_cache import PrefixCache, PrefixEntry

BLOCK = 16
MIN_PREFIX = 16


def _tokens(n: int, seed: int = 0) -> list[int]:
    return [((seed * 1000003) + i) % 50257 for i in range(n)]


def test_insert_then_find_longest_prefix_full_length() -> None:
    cache = PrefixCache(max_entries=32, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    toks = _tokens(64, seed=1)
    cache.insert(toks, kv_ref="kv-A")
    length, entry = cache.find_longest_prefix(toks)
    assert length == 64
    assert entry is not None
    assert entry.kv_ref == "kv-A"
    assert entry.prefix_length == 64


def test_partial_prefix_match_returns_partial_length() -> None:
    cache = PrefixCache(max_entries=32, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    shared = _tokens(32, seed=2)
    full = shared + _tokens(32, seed=999)
    cache.insert(full, kv_ref="kv-full")
    # Query shares only the first 32 tokens, then diverges.
    query = shared + [1234567 for _ in range(32)]
    length, entry = cache.find_longest_prefix(query)
    assert length == 32
    assert entry is not None
    assert entry.prefix_length == 32


def test_no_match_returns_zero_none() -> None:
    cache = PrefixCache(max_entries=32, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    cache.insert(_tokens(32, seed=3), kv_ref="kv")
    length, entry = cache.find_longest_prefix(_tokens(32, seed=4))
    assert length == 0
    assert entry is None


def test_min_prefix_tokens_enforced() -> None:
    # min=32 so a 16-token match is below threshold and rejected.
    cache = PrefixCache(max_entries=8, min_prefix_tokens=32, block_size=BLOCK)
    toks = _tokens(16, seed=5)
    cache.insert(toks, kv_ref="kv")
    length, entry = cache.find_longest_prefix(toks + _tokens(16, seed=6))
    assert length == 0
    assert entry is None


def test_lru_eviction_when_max_entries_exceeded() -> None:
    cache = PrefixCache(max_entries=2, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    # Each insert of 16 tokens adds exactly one entry.
    cache.insert(_tokens(16, seed=10), kv_ref="A")
    cache.insert(_tokens(16, seed=11), kv_ref="B")
    cache.insert(_tokens(16, seed=12), kv_ref="C")
    # max_entries=2 so one was evicted; oldest is 'A'.
    assert len(cache) == 2
    length, _ = cache.find_longest_prefix(_tokens(16, seed=10))
    assert length == 0  # 'A' evicted
    length_b, _ = cache.find_longest_prefix(_tokens(16, seed=11))
    length_c, _ = cache.find_longest_prefix(_tokens(16, seed=12))
    assert length_b == 16
    assert length_c == 16


def test_evict_lru_returns_the_evicted_entry() -> None:
    cache = PrefixCache(max_entries=4, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    cache.insert(_tokens(16, seed=20), kv_ref="A")
    time.sleep(0.001)
    cache.insert(_tokens(16, seed=21), kv_ref="B")
    victim = cache.evict_lru()
    assert isinstance(victim, PrefixEntry)
    assert victim.kv_ref == "A"
    assert len(cache) == 1


def test_refcount_protects_from_eviction() -> None:
    cache = PrefixCache(max_entries=2, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    cache.insert(_tokens(16, seed=30), kv_ref="A")
    # Pin A.
    _, entry_a = cache.find_longest_prefix(_tokens(16, seed=30))
    assert entry_a is not None
    entry_a.refcount = 1
    cache.insert(_tokens(16, seed=31), kv_ref="B")
    cache.insert(_tokens(16, seed=32), kv_ref="C")
    # A must survive; B (oldest unpinned) should be evicted.
    length_a, _ = cache.find_longest_prefix(_tokens(16, seed=30))
    length_b, _ = cache.find_longest_prefix(_tokens(16, seed=31))
    length_c, _ = cache.find_longest_prefix(_tokens(16, seed=32))
    assert length_a == 16
    assert length_b == 0
    assert length_c == 16


def test_stats_counts_correct() -> None:
    cache = PrefixCache(max_entries=4, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    cache.insert(_tokens(16, seed=40), kv_ref="A")
    cache.find_longest_prefix(_tokens(16, seed=40))  # hit
    cache.find_longest_prefix(_tokens(16, seed=41))  # miss
    s = cache.stats()
    assert s["entries"] == 1
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["inserts"] == 1
    assert s["max_entries"] == 4
    assert s["block_size"] == BLOCK


def test_duplicate_insert_does_not_double_store() -> None:
    cache = PrefixCache(max_entries=8, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    toks = _tokens(32, seed=50)
    cache.insert(toks, kv_ref="A")
    n_before = len(cache)
    cache.insert(toks, kv_ref="A")
    cache.insert(toks, kv_ref="A")
    assert len(cache) == n_before  # no new entries
    # Full length still reachable.
    length, entry = cache.find_longest_prefix(toks)
    assert length == 32
    assert entry is not None


def test_determinism() -> None:
    # Two separate cache instances should produce identical tokens_hash
    # for the same input -- no reliance on Python's per-process hash salt.
    cache_a = PrefixCache(max_entries=4, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    cache_b = PrefixCache(max_entries=4, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    toks = _tokens(32, seed=60)
    cache_a.insert(toks, kv_ref="A")
    cache_b.insert(toks, kv_ref="A")
    _, entry_a = cache_a.find_longest_prefix(toks)
    _, entry_b = cache_b.find_longest_prefix(toks)
    assert entry_a is not None and entry_b is not None
    assert entry_a.tokens_hash == entry_b.tokens_hash


def test_block_size_validation() -> None:
    with pytest.raises(ValueError):
        PrefixCache(block_size=0)
    with pytest.raises(ValueError):
        PrefixCache(block_size=-4)
    with pytest.raises(ValueError):
        PrefixCache(block_size=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PrefixCache(max_entries=0)


def test_thousand_insertions_lookup_under_one_second() -> None:
    # 1000 insertions * 4 block-prefixes each = up to 4000 entries.
    cache = PrefixCache(max_entries=8000, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    corpus = [_tokens(64, seed=s) for s in range(1000)]

    t0 = time.perf_counter()
    for toks in corpus:
        cache.insert(toks, kv_ref=None)
    insert_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for toks in corpus:
        length, entry = cache.find_longest_prefix(toks)
        assert length == 64
        assert entry is not None
    lookup_elapsed = time.perf_counter() - t0

    assert lookup_elapsed < 1.0, (
        f"1000 lookups took {lookup_elapsed:.3f}s (insert took {insert_elapsed:.3f}s)"
    )


def test_block_aligned_matching_only() -> None:
    # Sharing unit is block_size; a query that shares <1 full block gets 0.
    cache = PrefixCache(max_entries=4, min_prefix_tokens=MIN_PREFIX, block_size=BLOCK)
    toks = _tokens(32, seed=70)
    cache.insert(toks, kv_ref="A")
    query = toks[:10] + _tokens(32, seed=999)  # only 10 shared tokens
    length, entry = cache.find_longest_prefix(query)
    assert length == 0
    assert entry is None
