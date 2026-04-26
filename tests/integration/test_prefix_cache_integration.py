"""Integration tests for prefix_cache registration and end-to-end use."""

from __future__ import annotations

from src import longcontext as lc
from src.longcontext.prefix_cache import PrefixCache, PrefixEntry

EXPECTED_PRIOR = [
    "kv_int8",
    "attention_sinks",
    "ring_attention",
    "context_compaction",
    "kv_kivi_int4",
    "infini",
    "chunked_prefill",
    "paged_kv",
]


def test_registry_contains_prefix_cache():
    assert "prefix_cache" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["prefix_cache"] is PrefixCache


def test_registry_prior_entries_intact():
    for name in EXPECTED_PRIOR:
        assert name in lc.LONGCONTEXT_STRATEGY_REGISTRY, (
            f"prior registry entry {name!r} was removed"
        )


def test_public_surface_exports_prefix_cache():
    assert PrefixCache is lc.PrefixCache
    assert PrefixEntry is lc.PrefixEntry


def test_end_to_end_insert_and_find():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["prefix_cache"]
    cache = cls(max_entries=16, min_prefix_tokens=16, block_size=16)
    tokens = list(range(64))
    cache.insert(tokens, kv_ref="kv-ref-xyz")
    length, entry = cache.find_longest_prefix(tokens + [9999, 9998])
    assert length == 64
    assert entry is not None
    assert entry.kv_ref == "kv-ref-xyz"
    stats = cache.stats()
    assert stats["entries"] >= 1
    assert stats["hits"] == 1
