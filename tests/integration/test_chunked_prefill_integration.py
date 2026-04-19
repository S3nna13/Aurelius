"""Integration tests for chunked prefill registry + end-to-end usage."""

from __future__ import annotations

import torch

from src import longcontext as lc
from src.longcontext import (
    LONGCONTEXT_STRATEGY_REGISTRY,
    ChunkedPrefill,
    ChunkedPrefillConfig,
)


def test_registry_contains_chunked_prefill():
    assert "chunked_prefill" in LONGCONTEXT_STRATEGY_REGISTRY
    assert LONGCONTEXT_STRATEGY_REGISTRY["chunked_prefill"] is ChunkedPrefill


def test_registry_prior_entries_intact():
    for key in (
        "kv_int8",
        "attention_sinks",
        "ring_attention",
        "context_compaction",
        "kv_kivi_int4",
        "infini",
    ):
        assert key in LONGCONTEXT_STRATEGY_REGISTRY, key


def test_package_exports_chunked_prefill_symbols():
    assert hasattr(lc, "ChunkedPrefill")
    assert hasattr(lc, "ChunkedPrefillConfig")


def test_end_to_end_toy_chunk_fn():
    """Simulate a prefill pass: embed-like transform applied chunk by chunk."""
    torch.manual_seed(0)
    B, S, chunk_size = 2, 40, 8
    vocab = 37
    hidden = 6

    embed = torch.nn.Embedding(vocab, hidden)
    input_ids = torch.randint(0, vocab, (B, S))

    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=chunk_size, overlap=0))

    def chunk_fn(chunk_ids: torch.Tensor) -> torch.Tensor:
        return embed(chunk_ids)

    chunked_out = sched.run_chunk_fn(input_ids, chunk_fn=chunk_fn, concat_dim=1)
    full_out = embed(input_ids)

    assert chunked_out.shape == full_out.shape
    assert torch.allclose(chunked_out, full_out)


def test_end_to_end_with_overlap_preserves_stride():
    B, S = 2, 20
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=8, overlap=2))
    ids = torch.arange(B * S).reshape(B, S)
    chunks = list(sched.iter_chunks(ids))
    # First chunk starts at 0, each subsequent advances by stride=6.
    assert chunks[0][0] == 0
    assert chunks[1][0] == 6
    assert chunks[-1][1] == S
