"""Integration tests for paged_kv registration and end-to-end use."""

from __future__ import annotations

import torch

from src import longcontext as lc
from src.longcontext.paged_kv_cache import PagedKVCache


EXPECTED_PRIOR = [
    "kv_int8",
    "attention_sinks",
    "ring_attention",
    "context_compaction",
    "kv_kivi_int4",
    "infini",
    "chunked_prefill",
]


def test_registry_contains_paged_kv():
    assert "paged_kv" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["paged_kv"] is PagedKVCache


def test_registry_prior_entries_intact():
    for name in EXPECTED_PRIOR:
        assert name in lc.LONGCONTEXT_STRATEGY_REGISTRY, (
            f"prior registry entry {name!r} was removed"
        )


def test_end_to_end_allocate_write_read():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["paged_kv"]
    cache = cls(n_heads=2, head_dim=8, page_size=4, num_pages=8)
    table = cache.allocate("req", 10)  # 3 pages
    assert len(table.logical_pages) == 3
    g = torch.Generator().manual_seed(42)
    tokens = []
    for pos in range(10):
        k = torch.randn(2, 8, generator=g)
        v = torch.randn(2, 8, generator=g)
        cache.write("req", pos, k, v)
        tokens.append((k, v))
    out_k, out_v = cache.read("req", 0, 10)
    assert out_k.shape == (10, 2, 8)
    for i in range(10):
        assert torch.allclose(out_k[i], tokens[i][0])
        assert torch.allclose(out_v[i], tokens[i][1])
    cache.deallocate("req")
    assert cache.num_free_pages() == 8
