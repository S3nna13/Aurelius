"""Integration test for ChunkPrefillScheduler.

Scenario
--------
Three requests with different prompt lengths are added and stepped through
``schedule_batch`` until every request is in the decoding phase.  The test
verifies that:

* ``prefill_done`` matches the full prompt length for each request.
* Decode slots appear in batches once a request transitions.
* The DECODER_REGISTRY is wired with the ``"chunk_prefill"`` key.
"""

from __future__ import annotations

import pytest

from src.inference.chunk_prefill_scheduler import (
    ChunkPrefillConfig,
    ChunkPrefillScheduler,
    Request,
)
from src.inference import DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_decoder_registry_wired():
    assert "chunk_prefill" in DECODER_REGISTRY
    assert DECODER_REGISTRY["chunk_prefill"] is ChunkPrefillScheduler


# ---------------------------------------------------------------------------
# Full prefill-to-decode pipeline
# ---------------------------------------------------------------------------

def test_full_prefill_to_decode_pipeline():
    """Step through schedule_batch until all 3 requests are decoding."""
    cfg = ChunkPrefillConfig(
        chunk_size=60,
        max_prefill_tokens_per_batch=200,
        max_decode_tokens_per_batch=32,
        max_batch_size=32,
    )
    sched = ChunkPrefillScheduler(cfg)

    # Requests: prompt lengths 100, 200, 50.
    sched.add_request(Request("r1", prompt_tokens=100, max_output_tokens=20))
    sched.add_request(Request("r2", prompt_tokens=200, max_output_tokens=20))
    sched.add_request(Request("r3", prompt_tokens=50, max_output_tokens=20))

    # Step until all are decoding (max-guard to prevent infinite loop).
    MAX_STEPS = 100
    for step in range(MAX_STEPS):
        if sched.decoding_requests() == 3:
            break
        sched.schedule_batch()
    else:
        pytest.fail("All requests did not reach decoding within max steps")

    # Verify prefill_done matches full prompt lengths.
    assert sched._states["r1"].prefill_done == 100
    assert sched._states["r2"].prefill_done == 200
    assert sched._states["r3"].prefill_done == 50

    # All three should be decoding (none completed yet — max_output_tokens=20
    # and we haven't driven enough decode steps to exhaust them).
    assert sched.decoding_requests() == 3
    assert sched.prefilling_requests() == 0
    assert sched.pending_requests() == 3


def test_decode_slots_appear_after_prefill():
    """Decode slots must appear in batches once a request transitions."""
    cfg = ChunkPrefillConfig(
        chunk_size=60,
        max_prefill_tokens_per_batch=200,
        max_decode_tokens_per_batch=32,
        max_batch_size=32,
    )
    sched = ChunkPrefillScheduler(cfg)
    sched.add_request(Request("r1", prompt_tokens=50, max_output_tokens=20))
    sched.add_request(Request("r2", prompt_tokens=200, max_output_tokens=20))

    decode_seen = False
    for _ in range(50):
        batch = sched.schedule_batch()
        for slot in batch:
            if slot.slot_type == "decode":
                decode_seen = True
        if decode_seen:
            break

    assert decode_seen, "Expected decode slots to appear but none were observed"


def test_complete_all_via_complete_decode():
    """complete_decode drives all requests to completed state."""
    cfg = ChunkPrefillConfig(chunk_size=512)
    sched = ChunkPrefillScheduler(cfg)
    sched.add_request(Request("r1", prompt_tokens=50, max_output_tokens=50))
    sched.add_request(Request("r2", prompt_tokens=30, max_output_tokens=50))

    # Drive both through prefill.
    for _ in range(10):
        sched.schedule_batch()
        if sched.prefilling_requests() == 0:
            break

    # Both should be decoding now.
    assert sched.decoding_requests() == 2

    sched.complete_decode("r1")
    sched.complete_decode("r2")

    assert sched.completed_requests() == 2
    assert sched.pending_requests() == 0
