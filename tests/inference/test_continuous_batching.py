"""Tests for continuous batching scheduler."""

import pytest

from src.inference.continuous_batching import (
    BatchStep,
    ContinuousBatchScheduler,
    GenerationRequest,
)


def make_request(request_id: str, prompt_tokens: int = 4, max_new_tokens: int = 3) -> GenerationRequest:
    return GenerationRequest(request_id=request_id, prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens)


def test_schedule_prefill_respects_batch_size():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_prefill_tokens=20)
    scheduler.add_request(make_request("a"))
    scheduler.add_request(make_request("b"))
    scheduler.add_request(make_request("c"))
    step = scheduler.schedule_step()
    assert isinstance(step, BatchStep)
    assert len(step.all_ids) == 2


def test_schedule_prefill_respects_token_budget():
    scheduler = ContinuousBatchScheduler(max_batch_size=3, max_prefill_tokens=5)
    scheduler.add_request(make_request("a", prompt_tokens=4))
    scheduler.add_request(make_request("b", prompt_tokens=4))
    step = scheduler.schedule_step()
    assert step.prefill_ids == ["a"]


def test_decode_requests_are_rescheduled_round_robin():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_prefill_tokens=20)
    scheduler.add_request(make_request("a"))
    scheduler.add_request(make_request("b"))
    scheduler.schedule_step()
    step = scheduler.schedule_step()
    assert step.decode_ids == ["a", "b"]


def test_mark_step_complete_finishes_requests():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_prefill_tokens=20)
    scheduler.add_request(make_request("a", max_new_tokens=1))
    scheduler.schedule_step()
    scheduler.mark_step_complete(["a"], generated_tokens=1)
    assert scheduler.active_request_ids() == []


def test_has_pending_reflects_queue_state():
    scheduler = ContinuousBatchScheduler(max_batch_size=1, max_prefill_tokens=20)
    assert not scheduler.has_pending()
    scheduler.add_request(make_request("a", max_new_tokens=1))
    assert scheduler.has_pending()
    scheduler.schedule_step()
    scheduler.mark_step_complete(["a"], generated_tokens=1)
    assert not scheduler.has_pending()


def test_duplicate_request_ids_are_rejected():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_prefill_tokens=20)
    scheduler.add_request(make_request("dup"))
    with pytest.raises(ValueError):
        scheduler.add_request(make_request("dup"))


def test_invalid_scheduler_args_are_rejected():
    with pytest.raises(ValueError):
        ContinuousBatchScheduler(max_batch_size=0, max_prefill_tokens=10)
    with pytest.raises(ValueError):
        ContinuousBatchScheduler(max_batch_size=1, max_prefill_tokens=0)
