"""Integration test for the Orca-style continuous batching scheduler.

Drives the scheduler with a fake ``step_fn`` that returns ``input_ids[-1] + 1``
for each active request, verifying that FIFO admission, iteration-level
batching, and completion all compose end-to-end.
"""

from __future__ import annotations

import pytest

from src.inference import (
    SCHEDULER_REGISTRY,
    BatchStep,
    ContinuousBatchingScheduler,
    InferenceRequest,
)


def _fake_step_fn(step: BatchStep) -> dict:
    """Return a token that is ``last_input_token + 1`` per request."""
    out = {}
    for rid, ids in zip(step.request_ids, step.input_ids):
        out[rid] = ids[-1] + 1
    return out


def test_scheduler_registered_in_package():
    assert "continuous_batching" in SCHEDULER_REGISTRY
    assert SCHEDULER_REGISTRY["continuous_batching"] is ContinuousBatchingScheduler


def test_four_requests_two_slots_all_complete():
    sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=128)

    specs = [
        ("r0", [10], 3),
        ("r1", [20], 4),
        ("r2", [30], 2),
        ("r3", [40], 5),
    ]
    for rid, prompt, max_new in specs:
        sched.enqueue(
            InferenceRequest(
                request_id=rid,
                prompt_tokens=list(prompt),
                max_new_tokens=max_new,
                eos_token_id=-1,  # unreachable so max_new_tokens governs completion
            )
        )

    finished = {}
    # Bound iterations defensively; should finish well before this.
    for _ in range(200):
        step = sched.build_step()
        if step is None:
            break

        # Batch must never exceed max_batch_size.
        assert len(step.request_ids) <= 2

        tokens = _fake_step_fn(step)
        sched.receive_tokens(tokens)
        for req in sched.completed():
            finished[req.request_id] = req
    else:
        pytest.fail("scheduler did not drain within iteration budget")

    assert set(finished) == {"r0", "r1", "r2", "r3"}
    assert sched.stats() == {"active": 0, "queued": 0, "completed": 0}

    # Each request should have generated exactly max_new_tokens tokens,
    # and the first generated token should be prompt[-1] + 1.
    for rid, prompt, max_new in specs:
        req = finished[rid]
        assert len(req.generated) == max_new, (rid, req.generated)
        assert req.generated[0] == prompt[-1] + 1
        # Subsequent tokens each increment by 1 (since we feed last token back).
        for i in range(1, len(req.generated)):
            assert req.generated[i] == req.generated[i - 1] + 1
        assert req.state == "completed"


def test_eos_short_circuits_in_integration():
    sched = ContinuousBatchingScheduler(max_batch_size=2)
    # If we set eos = prompt_last + 1, the very first generated token is eos.
    sched.enqueue(
        InferenceRequest(
            request_id="eosy",
            prompt_tokens=[41],
            max_new_tokens=100,
            eos_token_id=42,
        )
    )
    step = sched.build_step()
    sched.receive_tokens(_fake_step_fn(step))
    done = sched.completed()
    assert len(done) == 1
    assert done[0].generated == [42]
