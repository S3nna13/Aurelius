"""Unit tests for the Orca-style continuous batching scheduler."""

import pytest

from src.inference.continuous_batching_scheduler import (
    BatchStep,
    ContinuousBatchingScheduler,
    InferenceRequest,
)


def _req(rid, prompt=(1, 2, 3), max_new_tokens=8, eos=0):
    return InferenceRequest(
        request_id=rid,
        prompt_tokens=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
    )


def test_enqueue_then_build_step_is_prefill():
    sched = ContinuousBatchingScheduler(max_batch_size=4, max_seq_len=64)
    sched.enqueue(_req("a", prompt=[10, 11, 12]))

    step = sched.build_step()
    assert isinstance(step, BatchStep)
    assert step.request_ids == ["a"]
    assert step.input_ids == [[10, 11, 12]]
    assert step.is_prefill == [True]


def test_after_receive_tokens_next_step_is_decoding_with_last_token():
    sched = ContinuousBatchingScheduler(max_batch_size=4, max_seq_len=64)
    sched.enqueue(_req("a", prompt=[10, 11, 12]))

    sched.build_step()  # prefill
    sched.receive_tokens({"a": 99})

    step2 = sched.build_step()
    assert step2.request_ids == ["a"]
    assert step2.input_ids == [[99]]
    assert step2.is_prefill == [False]


def test_max_batch_size_limits_active_requests():
    sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=64)
    for rid in ("a", "b", "c"):
        sched.enqueue(_req(rid))

    step = sched.build_step()
    assert len(step.request_ids) == 2
    assert step.request_ids == ["a", "b"]
    assert sched.stats() == {"active": 2, "queued": 1, "completed": 0}


def test_eos_token_completes_request():
    sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=64)
    sched.enqueue(_req("a", eos=7))

    sched.build_step()
    sched.receive_tokens({"a": 7})

    assert sched.stats()["active"] == 0
    drained = sched.completed()
    assert [r.request_id for r in drained] == ["a"]
    assert drained[0].generated == [7]
    assert drained[0].state == "completed"


def test_max_new_tokens_completes_request():
    sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=64)
    sched.enqueue(_req("a", max_new_tokens=3, eos=999))

    sched.build_step()
    sched.receive_tokens({"a": 1})
    sched.build_step()
    sched.receive_tokens({"a": 2})
    sched.build_step()
    sched.receive_tokens({"a": 3})

    assert sched.stats()["active"] == 0
    drained = sched.completed()
    assert drained[0].generated == [1, 2, 3]


def test_max_seq_len_completes_request():
    sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=5)
    # prompt length 4, so after 2 generated tokens total=6 > 5 -> complete
    sched.enqueue(
        _req("a", prompt=[1, 2, 3, 4], max_new_tokens=100, eos=999)
    )

    sched.build_step()
    sched.receive_tokens({"a": 10})
    assert sched.stats()["active"] == 1  # 4+1=5, not >5
    sched.build_step()
    sched.receive_tokens({"a": 11})
    assert sched.stats()["active"] == 0
    drained = sched.completed()
    assert drained[0].generated == [10, 11]


def test_empty_queue_and_no_active_returns_none():
    sched = ContinuousBatchingScheduler()
    assert sched.build_step() is None


def test_duplicate_enqueue_raises():
    sched = ContinuousBatchingScheduler()
    sched.enqueue(_req("a"))
    with pytest.raises(ValueError):
        sched.enqueue(_req("a"))

    # Even after completion, the id remains known and cannot be re-enqueued.
    sched.build_step()
    sched.receive_tokens({"a": 0})  # eos=0 completes
    sched.completed()
    with pytest.raises(ValueError):
        sched.enqueue(_req("a"))


def test_fifo_admission_order():
    sched = ContinuousBatchingScheduler(max_batch_size=2)
    for rid in ("first", "second", "third", "fourth"):
        sched.enqueue(_req(rid))

    step = sched.build_step()
    assert step.request_ids == ["first", "second"]

    # complete "first"
    sched.receive_tokens({"first": 0, "second": 42})  # eos=0
    sched.completed()

    # Next step admits "third" (FIFO) alongside still-active "second"
    step2 = sched.build_step()
    assert set(step2.request_ids) == {"second", "third"}
    # third is prefill, second is decoding
    for rid, is_p in zip(step2.request_ids, step2.is_prefill):
        if rid == "third":
            assert is_p is True
        else:
            assert is_p is False


def test_receive_tokens_unknown_id_raises():
    sched = ContinuousBatchingScheduler()
    sched.enqueue(_req("a"))
    sched.build_step()
    with pytest.raises(KeyError):
        sched.receive_tokens({"ghost": 5})

    # And no mutation happened to "a".
    assert sched._active["a"].generated == []


def test_empty_prompt_tokens_raises():
    sched = ContinuousBatchingScheduler()
    with pytest.raises(ValueError):
        sched.enqueue(
            InferenceRequest(
                request_id="a",
                prompt_tokens=[],
                max_new_tokens=4,
                eos_token_id=0,
            )
        )


def test_stats_counts_are_correct():
    sched = ContinuousBatchingScheduler(max_batch_size=2)
    for rid in ("a", "b", "c"):
        sched.enqueue(_req(rid, eos=0))

    assert sched.stats() == {"active": 0, "queued": 3, "completed": 0}

    sched.build_step()
    assert sched.stats() == {"active": 2, "queued": 1, "completed": 0}

    sched.receive_tokens({"a": 0, "b": 1})  # a hits eos
    assert sched.stats() == {"active": 1, "queued": 1, "completed": 1}

    sched.completed()
    assert sched.stats() == {"active": 1, "queued": 1, "completed": 0}


def test_determinism_same_inputs_yield_same_batches():
    def run():
        sched = ContinuousBatchingScheduler(max_batch_size=2, max_seq_len=32)
        for rid in ("a", "b", "c"):
            sched.enqueue(_req(rid, prompt=[1, 2, 3], max_new_tokens=4, eos=999))
        trace = []
        for _ in range(6):
            step = sched.build_step()
            if step is None:
                trace.append(None)
                break
            trace.append((tuple(step.request_ids), tuple(tuple(x) for x in step.input_ids), tuple(step.is_prefill)))
            # deterministic token: id-based
            tokens = {rid: (i + 1) for i, rid in enumerate(step.request_ids)}
            sched.receive_tokens(tokens)
        return trace

    assert run() == run()


def test_batchstep_shape_consistency():
    sched = ContinuousBatchingScheduler(max_batch_size=3)
    for rid in ("a", "b"):
        sched.enqueue(_req(rid, prompt=[1, 2]))
    step = sched.build_step()
    assert len(step.request_ids) == len(step.input_ids) == len(step.is_prefill)
