"""Tests for src/inference/continuous_batching_v2.py (~50 tests)."""

from __future__ import annotations

import pytest

from src.inference.continuous_batching_v2 import (
    BatchRequest,
    BatchingConfig,
    ContinuousBatcherV2,
    RequestPriority,
)


# ---------------------------------------------------------------------------
# RequestPriority enum
# ---------------------------------------------------------------------------

class TestRequestPriority:
    def test_critical_value(self):
        assert RequestPriority.CRITICAL == "critical"

    def test_high_value(self):
        assert RequestPriority.HIGH == "high"

    def test_normal_value(self):
        assert RequestPriority.NORMAL == "normal"

    def test_low_value(self):
        assert RequestPriority.LOW == "low"

    def test_four_members(self):
        assert len(RequestPriority) == 4

    def test_is_str_enum(self):
        assert isinstance(RequestPriority.NORMAL, str)


# ---------------------------------------------------------------------------
# BatchRequest
# ---------------------------------------------------------------------------

class TestBatchRequest:
    def test_auto_generates_id(self):
        req = BatchRequest(prompt_tokens=10, max_new_tokens=20)
        assert req.request_id is not None
        assert isinstance(req.request_id, str)

    def test_auto_generated_id_length(self):
        req = BatchRequest(prompt_tokens=10, max_new_tokens=20)
        assert len(req.request_id) == 8

    def test_two_requests_have_different_ids(self):
        r1 = BatchRequest(prompt_tokens=10, max_new_tokens=10)
        r2 = BatchRequest(prompt_tokens=10, max_new_tokens=10)
        assert r1.request_id != r2.request_id

    def test_priority_default(self):
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        assert req.priority == RequestPriority.NORMAL

    def test_generated_tokens_default(self):
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        assert req.generated_tokens == 0

    def test_custom_priority(self):
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.HIGH)
        assert req.priority == RequestPriority.HIGH

    def test_prompt_tokens_set(self):
        req = BatchRequest(prompt_tokens=100, max_new_tokens=50)
        assert req.prompt_tokens == 100

    def test_max_new_tokens_set(self):
        req = BatchRequest(prompt_tokens=100, max_new_tokens=50)
        assert req.max_new_tokens == 50

    def test_custom_id_override(self):
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, request_id="myid1234")
        assert req.request_id == "myid1234"


# ---------------------------------------------------------------------------
# BatchingConfig
# ---------------------------------------------------------------------------

class TestBatchingConfig:
    def test_max_batch_tokens_default(self):
        cfg = BatchingConfig()
        assert cfg.max_batch_tokens == 4096

    def test_max_batch_size_default(self):
        cfg = BatchingConfig()
        assert cfg.max_batch_size == 32

    def test_preemption_enabled_default(self):
        cfg = BatchingConfig()
        assert cfg.preemption_enabled is True

    def test_custom_max_batch_tokens(self):
        cfg = BatchingConfig(max_batch_tokens=2048)
        assert cfg.max_batch_tokens == 2048

    def test_custom_max_batch_size(self):
        cfg = BatchingConfig(max_batch_size=8)
        assert cfg.max_batch_size == 8

    def test_preemption_disabled(self):
        cfg = BatchingConfig(preemption_enabled=False)
        assert cfg.preemption_enabled is False


# ---------------------------------------------------------------------------
# ContinuousBatcherV2 construction
# ---------------------------------------------------------------------------

class TestContinuousBatcherV2Construction:
    def test_default_config_created(self):
        batcher = ContinuousBatcherV2()
        assert isinstance(batcher.config, BatchingConfig)

    def test_custom_config_used(self):
        cfg = BatchingConfig(max_batch_size=4)
        batcher = ContinuousBatcherV2(config=cfg)
        assert batcher.config.max_batch_size == 4

    def test_none_config_uses_defaults(self):
        batcher = ContinuousBatcherV2(config=None)
        assert batcher.config.max_batch_tokens == 4096

    def test_initial_queue_depth_zero(self):
        batcher = ContinuousBatcherV2()
        assert batcher.queue_depth() == 0


# ---------------------------------------------------------------------------
# enqueue / queue_depth
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_enqueue_increases_depth(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=10, max_new_tokens=10)
        batcher.enqueue(req)
        assert batcher.queue_depth() == 1

    def test_enqueue_multiple(self):
        batcher = ContinuousBatcherV2()
        for _ in range(5):
            batcher.enqueue(BatchRequest(prompt_tokens=5, max_new_tokens=5))
        assert batcher.queue_depth() == 5

    def test_queue_depth_returns_int(self):
        batcher = ContinuousBatcherV2()
        assert isinstance(batcher.queue_depth(), int)


# ---------------------------------------------------------------------------
# next_batch
# ---------------------------------------------------------------------------

class TestNextBatch:
    def test_returns_list(self):
        batcher = ContinuousBatcherV2()
        batcher.enqueue(BatchRequest(prompt_tokens=10, max_new_tokens=10))
        assert isinstance(batcher.next_batch(), list)

    def test_empty_queue_returns_empty(self):
        batcher = ContinuousBatcherV2()
        assert batcher.next_batch() == []

    def test_respects_max_batch_size(self):
        cfg = BatchingConfig(max_batch_size=3, max_batch_tokens=99999)
        batcher = ContinuousBatcherV2(config=cfg)
        for _ in range(10):
            batcher.enqueue(BatchRequest(prompt_tokens=1, max_new_tokens=1))
        batch = batcher.next_batch()
        assert len(batch) <= 3

    def test_respects_token_budget(self):
        cfg = BatchingConfig(max_batch_tokens=50, max_batch_size=99)
        batcher = ContinuousBatcherV2(config=cfg)
        for _ in range(10):
            batcher.enqueue(BatchRequest(prompt_tokens=15, max_new_tokens=0))
        batch = batcher.next_batch()
        total = sum(r.prompt_tokens + r.generated_tokens for r in batch)
        assert total <= 50

    def test_critical_before_low(self):
        cfg = BatchingConfig(max_batch_size=1, max_batch_tokens=99999)
        batcher = ContinuousBatcherV2(config=cfg)
        low_req = BatchRequest(prompt_tokens=10, max_new_tokens=10, priority=RequestPriority.LOW)
        critical_req = BatchRequest(prompt_tokens=10, max_new_tokens=10, priority=RequestPriority.CRITICAL)
        batcher.enqueue(low_req)
        batcher.enqueue(critical_req)
        batch = batcher.next_batch()
        assert len(batch) == 1
        assert batch[0].priority == RequestPriority.CRITICAL

    def test_high_before_normal(self):
        cfg = BatchingConfig(max_batch_size=1, max_batch_tokens=99999)
        batcher = ContinuousBatcherV2(config=cfg)
        normal_req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.NORMAL)
        high_req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.HIGH)
        batcher.enqueue(normal_req)
        batcher.enqueue(high_req)
        batch = batcher.next_batch()
        assert batch[0].priority == RequestPriority.HIGH

    def test_batch_requests_are_batch_request_instances(self):
        batcher = ContinuousBatcherV2()
        batcher.enqueue(BatchRequest(prompt_tokens=5, max_new_tokens=5))
        batch = batcher.next_batch()
        assert all(isinstance(r, BatchRequest) for r in batch)

    def test_next_batch_does_not_remove_from_queue(self):
        batcher = ContinuousBatcherV2()
        batcher.enqueue(BatchRequest(prompt_tokens=5, max_new_tokens=5))
        batcher.next_batch()
        assert batcher.queue_depth() == 1


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------

class TestMarkDone:
    def test_returns_true_for_valid_id(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        batcher.enqueue(req)
        assert batcher.mark_done(req.request_id) is True

    def test_returns_false_for_unknown_id(self):
        batcher = ContinuousBatcherV2()
        assert batcher.mark_done("nonexistent") is False

    def test_decreases_queue_depth(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        batcher.enqueue(req)
        batcher.mark_done(req.request_id)
        assert batcher.queue_depth() == 0

    def test_only_removes_target(self):
        batcher = ContinuousBatcherV2()
        r1 = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        r2 = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        batcher.enqueue(r1)
        batcher.enqueue(r2)
        batcher.mark_done(r1.request_id)
        assert batcher.queue_depth() == 1

    def test_double_done_returns_false(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5)
        batcher.enqueue(req)
        batcher.mark_done(req.request_id)
        assert batcher.mark_done(req.request_id) is False


# ---------------------------------------------------------------------------
# preempt_lowest
# ---------------------------------------------------------------------------

class TestPreemptLowest:
    def test_preemption_enabled_removes_from_queue(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.LOW)
        batcher.enqueue(req)
        removed = batcher.preempt_lowest(1)
        assert len(removed) == 1
        assert batcher.queue_depth() == 0

    def test_preemption_disabled_returns_empty(self):
        cfg = BatchingConfig(preemption_enabled=False)
        batcher = ContinuousBatcherV2(config=cfg)
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.LOW)
        batcher.enqueue(req)
        removed = batcher.preempt_lowest(1)
        assert removed == []
        assert batcher.queue_depth() == 1

    def test_preempt_lowest_priority_removed(self):
        batcher = ContinuousBatcherV2()
        critical = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.CRITICAL)
        low = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.LOW)
        batcher.enqueue(critical)
        batcher.enqueue(low)
        removed = batcher.preempt_lowest(1)
        assert removed[0].priority == RequestPriority.LOW

    def test_preempt_n_requests(self):
        batcher = ContinuousBatcherV2()
        for _ in range(5):
            batcher.enqueue(BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.LOW))
        removed = batcher.preempt_lowest(3)
        assert len(removed) == 3
        assert batcher.queue_depth() == 2

    def test_preempt_returns_list(self):
        batcher = ContinuousBatcherV2()
        batcher.enqueue(BatchRequest(prompt_tokens=5, max_new_tokens=5))
        result = batcher.preempt_lowest(1)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# token_utilization
# ---------------------------------------------------------------------------

class TestTokenUtilization:
    def test_empty_batch(self):
        batcher = ContinuousBatcherV2()
        assert batcher.token_utilization([]) == pytest.approx(0.0)

    def test_full_batch(self):
        cfg = BatchingConfig(max_batch_tokens=100)
        batcher = ContinuousBatcherV2(config=cfg)
        req = BatchRequest(prompt_tokens=100, max_new_tokens=0)
        assert batcher.token_utilization([req]) == pytest.approx(1.0)

    def test_half_utilization(self):
        cfg = BatchingConfig(max_batch_tokens=100)
        batcher = ContinuousBatcherV2(config=cfg)
        req = BatchRequest(prompt_tokens=50, max_new_tokens=0)
        assert batcher.token_utilization([req]) == pytest.approx(0.5)

    def test_utilization_includes_generated_tokens(self):
        cfg = BatchingConfig(max_batch_tokens=100)
        batcher = ContinuousBatcherV2(config=cfg)
        req = BatchRequest(prompt_tokens=40, max_new_tokens=100, generated_tokens=60)
        assert batcher.token_utilization([req]) == pytest.approx(1.0)

    def test_multi_request_utilization(self):
        cfg = BatchingConfig(max_batch_tokens=200)
        batcher = ContinuousBatcherV2(config=cfg)
        r1 = BatchRequest(prompt_tokens=50, max_new_tokens=0)
        r2 = BatchRequest(prompt_tokens=50, max_new_tokens=0)
        assert batcher.token_utilization([r1, r2]) == pytest.approx(0.5)

    def test_returns_float(self):
        batcher = ContinuousBatcherV2()
        result = batcher.token_utilization([])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _priority_score
# ---------------------------------------------------------------------------

class TestPriorityScore:
    def test_critical_score_is_zero(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.CRITICAL)
        assert batcher._priority_score(req) == 0

    def test_high_score_is_one(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.HIGH)
        assert batcher._priority_score(req) == 1

    def test_normal_score_is_two(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.NORMAL)
        assert batcher._priority_score(req) == 2

    def test_low_score_is_three(self):
        batcher = ContinuousBatcherV2()
        req = BatchRequest(prompt_tokens=5, max_new_tokens=5, priority=RequestPriority.LOW)
        assert batcher._priority_score(req) == 3
