"""Unit tests for the ChunkPrefillScheduler (10–16 tests)."""

from __future__ import annotations

import pytest

from src.inference.chunk_prefill_scheduler import (
    BatchSlot,
    ChunkPrefillConfig,
    ChunkPrefillScheduler,
    Request,
    RequestState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(rid: str, prompt: int = 100, max_out: int = 10, priority: int = 0) -> Request:
    return Request(
        request_id=rid,
        prompt_tokens=prompt,
        max_output_tokens=max_out,
        priority=priority,
    )


def _scheduler(**kwargs) -> ChunkPrefillScheduler:
    return ChunkPrefillScheduler(ChunkPrefillConfig(**kwargs))


# ---------------------------------------------------------------------------
# Test 1 — config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ChunkPrefillConfig()
    assert cfg.chunk_size == 512
    assert cfg.max_batch_size == 32
    assert cfg.max_prefill_tokens_per_batch == 4096
    assert cfg.max_decode_tokens_per_batch == 32


# ---------------------------------------------------------------------------
# Test 2 — add_request increments pending count
# ---------------------------------------------------------------------------

def test_add_request():
    sched = ChunkPrefillScheduler()
    sched.add_request(_req("r1"))
    assert sched.pending_requests() == 1
    assert sched.prefilling_requests() == 1
    assert sched.decoding_requests() == 0
    assert sched.completed_requests() == 0


# ---------------------------------------------------------------------------
# Test 3 — duplicate request_id raises ValueError
# ---------------------------------------------------------------------------

def test_add_duplicate_raises():
    sched = ChunkPrefillScheduler()
    sched.add_request(_req("r1"))
    with pytest.raises(ValueError, match="duplicate"):
        sched.add_request(_req("r1"))


# ---------------------------------------------------------------------------
# Test 4 — new request produces a prefill slot
# ---------------------------------------------------------------------------

def test_schedule_prefill_only():
    sched = ChunkPrefillScheduler()
    sched.add_request(_req("r1", prompt=50))
    batch = sched.schedule_batch()
    assert len(batch) == 1
    assert batch[0].slot_type == "prefill"
    assert batch[0].request_id == "r1"
    assert batch[0].n_tokens == 50


# ---------------------------------------------------------------------------
# Test 5 — prefill_done advances after schedule
# ---------------------------------------------------------------------------

def test_schedule_advances_prefill_done():
    sched = _scheduler(chunk_size=30)
    sched.add_request(_req("r1", prompt=100))
    sched.schedule_batch()
    state = sched._states["r1"]
    assert state.prefill_done == 30


# ---------------------------------------------------------------------------
# Test 6 — completing prefill transitions status to decoding
# ---------------------------------------------------------------------------

def test_prefill_completes_transitions_to_decoding():
    sched = _scheduler(chunk_size=512)
    sched.add_request(_req("r1", prompt=50))
    sched.schedule_batch()  # 50 tokens < chunk_size → all prefilled in one shot
    state = sched._states["r1"]
    assert state.status == "decoding"
    assert sched.decoding_requests() == 1
    assert sched.prefilling_requests() == 0


# ---------------------------------------------------------------------------
# Test 7 — decoding request gets a decode slot
# ---------------------------------------------------------------------------

def test_decode_slot_in_batch():
    sched = _scheduler(chunk_size=512)
    sched.add_request(_req("r1", prompt=50, max_out=5))
    sched.schedule_batch()  # prefill done → now decoding
    batch = sched.schedule_batch()
    decode_slots = [s for s in batch if s.slot_type == "decode"]
    assert len(decode_slots) == 1
    assert decode_slots[0].request_id == "r1"
    assert decode_slots[0].n_tokens == 1


# ---------------------------------------------------------------------------
# Test 8 — decode slots are scheduled before prefill slots
# ---------------------------------------------------------------------------

def test_decode_priority_over_prefill():
    sched = _scheduler(chunk_size=512)
    # r_fast: small prompt, completes prefill immediately → decoding
    sched.add_request(_req("r_fast", prompt=10, max_out=5))
    sched.schedule_batch()  # r_fast → decoding
    # r_slow: added after, still prefilling
    sched.add_request(_req("r_slow", prompt=200, max_out=5))
    batch = sched.schedule_batch()
    # First slot must be the decode slot
    assert batch[0].slot_type == "decode"
    assert batch[0].request_id == "r_fast"
    # Second slot should be prefill for r_slow
    prefill_slots = [s for s in batch if s.slot_type == "prefill"]
    assert len(prefill_slots) == 1
    assert prefill_slots[0].request_id == "r_slow"


# ---------------------------------------------------------------------------
# Test 9 — chunk_size is respected across multiple batches
# ---------------------------------------------------------------------------

def test_chunk_size_respected():
    sched = _scheduler(chunk_size=100, max_prefill_tokens_per_batch=200)
    sched.add_request(_req("r1", prompt=300))
    # Batch 1: 100 tokens prefilled
    b1 = sched.schedule_batch()
    assert b1[0].n_tokens == 100
    assert sched._states["r1"].prefill_done == 100
    assert sched._states["r1"].status == "prefilling"
    # Batch 2: another 100 tokens
    b2 = sched.schedule_batch()
    assert b2[0].n_tokens == 100
    assert sched._states["r1"].prefill_done == 200
    assert sched._states["r1"].status == "prefilling"
    # Batch 3: final 100 tokens → done
    b3 = sched.schedule_batch()
    assert b3[0].n_tokens == 100
    assert sched._states["r1"].prefill_done == 300
    assert sched._states["r1"].status == "decoding"


# ---------------------------------------------------------------------------
# Test 10 — complete_decode marks request as completed
# ---------------------------------------------------------------------------

def test_complete_decode():
    sched = _scheduler(chunk_size=512)
    sched.add_request(_req("r1", prompt=50, max_out=100))
    sched.schedule_batch()  # prefill
    assert sched._states["r1"].status == "decoding"
    sched.complete_decode("r1")
    assert sched._states["r1"].status == "completed"
    assert sched.completed_requests() == 1


# ---------------------------------------------------------------------------
# Test 11 — pending count decreases after completion
# ---------------------------------------------------------------------------

def test_pending_count_decreases():
    sched = _scheduler(chunk_size=512)
    sched.add_request(_req("r1", prompt=10, max_out=2))
    assert sched.pending_requests() == 1
    sched.schedule_batch()  # prefill → decoding
    sched.complete_decode("r1")
    assert sched.pending_requests() == 0


# ---------------------------------------------------------------------------
# Test 12 — mixed batch: one prefilling + one decoding in same batch
# ---------------------------------------------------------------------------

def test_mixed_batch():
    sched = _scheduler(chunk_size=512)
    sched.add_request(_req("r_decode", prompt=20, max_out=5))
    sched.schedule_batch()  # r_decode → decoding
    sched.add_request(_req("r_prefill", prompt=200, max_out=5))
    batch = sched.schedule_batch()
    types = {s.slot_type for s in batch}
    assert "decode" in types
    assert "prefill" in types
    assert len(batch) == 2


# ---------------------------------------------------------------------------
# Test 13 — max_prefill_tokens_per_batch limits total prefill tokens
# ---------------------------------------------------------------------------

def test_max_prefill_tokens_per_batch():
    sched = _scheduler(chunk_size=512, max_prefill_tokens_per_batch=100)
    sched.add_request(_req("r1", prompt=200))
    sched.add_request(_req("r2", prompt=200))
    batch = sched.schedule_batch()
    total_prefill = sum(s.n_tokens for s in batch if s.slot_type == "prefill")
    assert total_prefill <= 100


# ---------------------------------------------------------------------------
# Test 14 — empty schedule with no requests returns empty list
# ---------------------------------------------------------------------------

def test_empty_schedule():
    sched = ChunkPrefillScheduler()
    batch = sched.schedule_batch()
    assert batch == []


# ---------------------------------------------------------------------------
# Test 15 — multiple requests scheduled in priority order
# ---------------------------------------------------------------------------

def test_multiple_requests_priority_order():
    sched = _scheduler(chunk_size=512, max_prefill_tokens_per_batch=2048)
    sched.add_request(_req("low", prompt=50, priority=10))
    sched.add_request(_req("high", prompt=50, priority=1))
    sched.add_request(_req("mid", prompt=50, priority=5))
    batch = sched.schedule_batch()
    request_ids = [s.request_id for s in batch]
    # highest priority (lowest value) should be first
    assert request_ids[0] == "high"
    assert request_ids[1] == "mid"
    assert request_ids[2] == "low"


# ---------------------------------------------------------------------------
# Test 16 — utilization tracks per-batch statistics
# ---------------------------------------------------------------------------

def test_utilization_tracking():
    sched = _scheduler(chunk_size=50)
    sched.add_request(_req("r1", prompt=50, max_out=3))
    sched.schedule_batch()  # prefill: 50 tokens, 0 decode
    sched.schedule_batch()  # decode: 0 prefill, 1 decode
    stats = sched.utilization()
    assert stats["prefill_tokens_per_batch"] == 25.0   # mean(50, 0)
    assert stats["decode_slots_per_batch"] == 0.5       # mean(0, 1)
