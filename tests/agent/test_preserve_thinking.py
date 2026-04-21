"""Unit tests for src/agent/preserve_thinking.py (15 tests)."""

from __future__ import annotations

import time

import pytest

from src.agent.preserve_thinking import (
    PreserveThinkingBuffer,
    PreserveThinkingConfig,
    ThinkingSnapshot,
)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = PreserveThinkingConfig()
    assert cfg.max_snapshots == 8
    assert cfg.max_tokens_per_snapshot == 4096
    assert cfg.use_summary is False
    assert cfg.max_prepend_tokens == 2048


# ---------------------------------------------------------------------------
# 2. test_add_snapshot_basic
# ---------------------------------------------------------------------------

def test_add_snapshot_basic():
    buf = PreserveThinkingBuffer()
    snap = buf.add_snapshot(turn_id=0, thinking_tokens=[1, 2, 3])
    assert len(buf) == 1
    snaps = buf.get_all_snapshots()
    assert len(snaps) == 1
    assert snaps[0] is snap
    assert snap.turn_id == 0
    assert snap.thinking_tokens == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3. test_add_snapshot_truncates
# ---------------------------------------------------------------------------

def test_add_snapshot_truncates():
    cfg = PreserveThinkingConfig(max_tokens_per_snapshot=4)
    buf = PreserveThinkingBuffer(cfg)
    tokens = list(range(10))
    snap = buf.add_snapshot(turn_id=0, thinking_tokens=tokens)
    assert snap.thinking_tokens == [0, 1, 2, 3]
    assert len(snap.thinking_tokens) == 4


# ---------------------------------------------------------------------------
# 4. test_ring_buffer_evicts
# ---------------------------------------------------------------------------

def test_ring_buffer_evicts():
    cfg = PreserveThinkingConfig(max_snapshots=3)
    buf = PreserveThinkingBuffer(cfg)
    for i in range(4):
        buf.add_snapshot(turn_id=i, thinking_tokens=[i])
    assert len(buf) == 3


# ---------------------------------------------------------------------------
# 5. test_eviction_count
# ---------------------------------------------------------------------------

def test_eviction_count():
    cfg = PreserveThinkingConfig(max_snapshots=3)
    buf = PreserveThinkingBuffer(cfg)
    assert buf.eviction_count() == 0
    for i in range(5):
        buf.add_snapshot(turn_id=i, thinking_tokens=[i])
    assert buf.eviction_count() == 2


# ---------------------------------------------------------------------------
# 6. test_get_prepend_tokens_full
# ---------------------------------------------------------------------------

def test_get_prepend_tokens_full():
    cfg = PreserveThinkingConfig(use_summary=False, max_prepend_tokens=100)
    buf = PreserveThinkingBuffer(cfg)
    buf.add_snapshot(turn_id=0, thinking_tokens=[10, 20, 30], summary_tokens=[99])
    result = buf.get_prepend_tokens()
    assert result == [10, 20, 30]


# ---------------------------------------------------------------------------
# 7. test_get_prepend_tokens_summary
# ---------------------------------------------------------------------------

def test_get_prepend_tokens_summary():
    cfg = PreserveThinkingConfig(use_summary=True, max_prepend_tokens=100)
    buf = PreserveThinkingBuffer(cfg)
    buf.add_snapshot(turn_id=0, thinking_tokens=[10, 20, 30], summary_tokens=[7, 8])
    result = buf.get_prepend_tokens()
    assert result == [7, 8]


# ---------------------------------------------------------------------------
# 8. test_get_prepend_tokens_respects_max
# ---------------------------------------------------------------------------

def test_get_prepend_tokens_respects_max():
    cfg = PreserveThinkingConfig(use_summary=False, max_prepend_tokens=10)
    buf = PreserveThinkingBuffer(cfg)
    long_tokens = list(range(50))
    buf.add_snapshot(turn_id=0, thinking_tokens=long_tokens)
    result = buf.get_prepend_tokens()
    assert result == list(range(10))
    assert len(result) == 10


# ---------------------------------------------------------------------------
# 9. test_get_prepend_empty_buffer
# ---------------------------------------------------------------------------

def test_get_prepend_empty_buffer():
    buf = PreserveThinkingBuffer()
    assert buf.get_prepend_tokens() == []


# ---------------------------------------------------------------------------
# 10. test_clear
# ---------------------------------------------------------------------------

def test_clear():
    buf = PreserveThinkingBuffer()
    for i in range(4):
        buf.add_snapshot(turn_id=i, thinking_tokens=[i, i + 1])
    buf.clear()
    assert len(buf) == 0
    assert buf.get_all_snapshots() == []
    assert buf.get_prepend_tokens() == []


# ---------------------------------------------------------------------------
# 11. test_oldest_first_order
# ---------------------------------------------------------------------------

def test_oldest_first_order():
    buf = PreserveThinkingBuffer()
    for i in range(5):
        buf.add_snapshot(turn_id=i, thinking_tokens=[i])
    snaps = buf.get_all_snapshots()
    turn_ids = [s.turn_id for s in snaps]
    assert turn_ids == sorted(turn_ids)
    assert turn_ids[0] == 0
    assert turn_ids[-1] == 4


# ---------------------------------------------------------------------------
# 12. test_metadata_preserved
# ---------------------------------------------------------------------------

def test_metadata_preserved():
    buf = PreserveThinkingBuffer()
    meta = {"model": "aurelius-1.4B", "step": 42, "tags": ["cot", "plan"]}
    snap = buf.add_snapshot(turn_id=0, thinking_tokens=[1], metadata=meta)
    assert snap.metadata == meta
    # Mutating the original dict must not affect stored snapshot.
    meta["extra"] = "surprise"
    assert "extra" not in snap.metadata


# ---------------------------------------------------------------------------
# 13. test_summary_none_falls_back
# ---------------------------------------------------------------------------

def test_summary_none_falls_back():
    cfg = PreserveThinkingConfig(use_summary=True, max_prepend_tokens=100)
    buf = PreserveThinkingBuffer(cfg)
    buf.add_snapshot(turn_id=0, thinking_tokens=[5, 6, 7], summary_tokens=None)
    result = buf.get_prepend_tokens()
    # summary_tokens is None → stored as [] → falls back to thinking_tokens
    assert result == [5, 6, 7]


# ---------------------------------------------------------------------------
# 14. test_determinism
# ---------------------------------------------------------------------------

def test_determinism():
    def run():
        cfg = PreserveThinkingConfig(use_summary=False, max_prepend_tokens=50)
        buf = PreserveThinkingBuffer(cfg)
        for i in range(5):
            buf.add_snapshot(turn_id=i, thinking_tokens=list(range(i, i + 10)))
        return buf.get_prepend_tokens()

    assert run() == run()


# ---------------------------------------------------------------------------
# 15. test_multiple_turns_most_recent_wins
# ---------------------------------------------------------------------------

def test_multiple_turns_most_recent_wins():
    cfg = PreserveThinkingConfig(use_summary=False, max_prepend_tokens=100)
    buf = PreserveThinkingBuffer(cfg)
    buf.add_snapshot(turn_id=0, thinking_tokens=[1, 2])
    buf.add_snapshot(turn_id=1, thinking_tokens=[3, 4])
    buf.add_snapshot(turn_id=2, thinking_tokens=[5, 6])
    # Most recent is turn_id=2
    result = buf.get_prepend_tokens()
    assert result == [5, 6]
