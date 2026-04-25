"""Tests for src/longcontext/chunk_prefill.py (8+ tests)."""

from __future__ import annotations

import pytest

from src.longcontext.chunk_prefill import (
    ChunkPrefillConfig,
    ChunkPrefillScheduler,
    ChunkResult,
)

SCHEDULER = ChunkPrefillScheduler()
TOKENS = list(range(1000))


# ---------------------------------------------------------------------------
# 1. split: returns ChunkResult objects
# ---------------------------------------------------------------------------
def test_split_returns_chunk_results():
    chunks = SCHEDULER.split(TOKENS)
    assert all(isinstance(c, ChunkResult) for c in chunks)


# ---------------------------------------------------------------------------
# 2. split: last chunk marked is_last=True
# ---------------------------------------------------------------------------
def test_split_last_chunk_marked():
    chunks = SCHEDULER.split(TOKENS)
    assert chunks[-1].is_last is True
    for c in chunks[:-1]:
        assert c.is_last is False


# ---------------------------------------------------------------------------
# 3. split: chunk indices are sequential
# ---------------------------------------------------------------------------
def test_split_sequential_indices():
    chunks = SCHEDULER.split(TOKENS)
    for i, c in enumerate(chunks):
        assert c.chunk_idx == i


# ---------------------------------------------------------------------------
# 4. split: chunk sizes bounded by chunk_size
# ---------------------------------------------------------------------------
def test_split_chunk_size_bound():
    cfg = ChunkPrefillConfig(chunk_size=128, overlap=32)
    chunks = SCHEDULER.split(TOKENS, cfg)
    for c in chunks:
        assert len(c.token_ids) <= cfg.chunk_size


# ---------------------------------------------------------------------------
# 5. split: overlap preserved between consecutive chunks
# ---------------------------------------------------------------------------
def test_split_overlap():
    cfg = ChunkPrefillConfig(chunk_size=100, overlap=20)
    chunks = SCHEDULER.split(TOKENS, cfg)
    if len(chunks) > 1:
        # End of chunk[0] and start of chunk[1] should share tokens
        tail = chunks[0].token_ids[-cfg.overlap:]
        head = chunks[1].token_ids[:cfg.overlap]
        assert tail == head


# ---------------------------------------------------------------------------
# 6. split: max_chunks respected
# ---------------------------------------------------------------------------
def test_split_max_chunks():
    cfg = ChunkPrefillConfig(chunk_size=64, overlap=0, max_chunks=3)
    chunks = SCHEDULER.split(TOKENS, cfg)
    assert len(chunks) <= 3


# ---------------------------------------------------------------------------
# 7. split: invalid overlap raises ValueError
# ---------------------------------------------------------------------------
def test_split_invalid_overlap():
    cfg = ChunkPrefillConfig(chunk_size=64, overlap=64)
    with pytest.raises(ValueError):
        SCHEDULER.split(TOKENS, cfg)


# ---------------------------------------------------------------------------
# 8. estimate_memory keys present
# ---------------------------------------------------------------------------
def test_estimate_memory_keys():
    result = SCHEDULER.estimate_memory(TOKENS, n_layers=12, n_heads=8, head_dim=64)
    assert "n_chunks" in result
    assert "peak_kv_bytes" in result
    assert "total_kv_bytes" in result


# ---------------------------------------------------------------------------
# 9. estimate_memory: total >= peak
# ---------------------------------------------------------------------------
def test_estimate_memory_total_gte_peak():
    result = SCHEDULER.estimate_memory(TOKENS, n_layers=12, n_heads=8, head_dim=64)
    assert result["total_kv_bytes"] >= result["peak_kv_bytes"]


# ---------------------------------------------------------------------------
# 10. merge_kv_caches: filters None entries
# ---------------------------------------------------------------------------
def test_merge_kv_caches_filters_none():
    results = [
        ChunkResult(chunk_idx=0, token_ids=[1, 2], kv_cache={"k": [1]}, is_last=False),
        ChunkResult(chunk_idx=1, token_ids=[3, 4], kv_cache=None, is_last=False),
        ChunkResult(chunk_idx=2, token_ids=[5], kv_cache={"k": [2]}, is_last=True),
    ]
    merged = SCHEDULER.merge_kv_caches(results)
    assert len(merged) == 2
    assert all(m is not None for m in merged)


# ---------------------------------------------------------------------------
# 11. split: empty token list returns empty list
# ---------------------------------------------------------------------------
def test_split_empty():
    chunks = SCHEDULER.split([])
    assert chunks == []
