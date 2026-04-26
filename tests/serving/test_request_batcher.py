"""Tests for src/serving/request_batcher.py — ≥28 test cases."""

from __future__ import annotations

import time

import pytest

from src.serving.request_batcher import (
    REQUEST_BATCHER_REGISTRY,
    Batch,
    BatchRequest,
    RequestBatcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(rid: str = "r1", priority: int = 0, payload: dict | None = None) -> BatchRequest:
    return BatchRequest(request_id=rid, payload=payload or {}, priority=priority)


def _fill(batcher: RequestBatcher, n: int, priority: int = 0) -> list[BatchRequest]:
    reqs = [_req(f"req-{i}", priority=priority) for i in range(n)]
    for r in reqs:
        batcher.enqueue(r)
    return reqs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in REQUEST_BATCHER_REGISTRY

    def test_registry_default_is_class(self):
        assert REQUEST_BATCHER_REGISTRY["default"] is RequestBatcher


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------


class TestBatchRequestDataclass:
    def test_fields_stored(self):
        r = BatchRequest(request_id="x", payload={"k": "v"}, priority=3)
        assert r.request_id == "x"
        assert r.payload == {"k": "v"}
        assert r.priority == 3

    def test_enqueued_at_auto_set_when_zero(self):
        before = time.monotonic()
        r = BatchRequest(request_id="auto", payload={})
        after = time.monotonic()
        assert before <= r.enqueued_at <= after

    def test_enqueued_at_preserved_when_nonzero(self):
        ts = 12345.678
        r = BatchRequest(request_id="ts", payload={}, enqueued_at=ts)
        assert r.enqueued_at == ts

    def test_default_priority(self):
        r = _req()
        assert r.priority == 0


class TestBatchDataclass:
    def test_batch_is_frozen(self):
        b = Batch(batch_id="abc", requests=[], created_at=1.0)
        with pytest.raises((AttributeError, TypeError)):
            b.batch_id = "xyz"  # type: ignore[misc]

    def test_batch_fields(self):
        reqs = [_req()]
        b = Batch(batch_id="b1", requests=reqs, created_at=42.0)
        assert b.batch_id == "b1"
        assert b.requests is reqs
        assert b.created_at == 42.0


# ---------------------------------------------------------------------------
# enqueue / queue_size
# ---------------------------------------------------------------------------


class TestEnqueue:
    def test_enqueue_increments_size(self):
        batcher = RequestBatcher()
        assert batcher.queue_size() == 0
        batcher.enqueue(_req("r1"))
        assert batcher.queue_size() == 1
        batcher.enqueue(_req("r2"))
        assert batcher.queue_size() == 2

    def test_enqueue_does_not_form_batch_automatically(self):
        batcher = RequestBatcher(max_batch_size=2)
        batcher.enqueue(_req())
        # still below threshold, no side effects
        assert batcher.queue_size() == 1


# ---------------------------------------------------------------------------
# try_form_batch – insufficient queue
# ---------------------------------------------------------------------------


class TestTryFormBatchNone:
    def test_returns_none_when_empty(self):
        batcher = RequestBatcher()
        assert batcher.try_form_batch() is None

    def test_returns_none_below_max_batch_size_and_not_old(self):
        batcher = RequestBatcher(max_batch_size=4, max_wait_ms=100_000)
        _fill(batcher, 3)
        assert batcher.try_form_batch() is None

    def test_queue_unchanged_when_none_returned(self):
        batcher = RequestBatcher(max_batch_size=5, max_wait_ms=100_000)
        _fill(batcher, 3)
        batcher.try_form_batch()
        assert batcher.queue_size() == 3


# ---------------------------------------------------------------------------
# try_form_batch – batch at max_batch_size
# ---------------------------------------------------------------------------


class TestTryFormBatchAtCapacity:
    def test_returns_batch_at_max_batch_size(self):
        batcher = RequestBatcher(max_batch_size=4, max_wait_ms=100_000)
        _fill(batcher, 4)
        batch = batcher.try_form_batch()
        assert batch is not None

    def test_batch_contains_max_batch_size_requests(self):
        batcher = RequestBatcher(max_batch_size=3, max_wait_ms=100_000)
        _fill(batcher, 3)
        batch = batcher.try_form_batch()
        assert len(batch.requests) == 3  # type: ignore[union-attr]

    def test_queue_empty_after_exact_batch(self):
        batcher = RequestBatcher(max_batch_size=3, max_wait_ms=100_000)
        _fill(batcher, 3)
        batcher.try_form_batch()
        assert batcher.queue_size() == 0

    def test_queue_has_remainder_after_over_capacity(self):
        batcher = RequestBatcher(max_batch_size=3, max_wait_ms=100_000)
        _fill(batcher, 5)
        batcher.try_form_batch()
        assert batcher.queue_size() == 2

    def test_batch_id_is_string(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        _fill(batcher, 2)
        batch = batcher.try_form_batch()
        assert isinstance(batch.batch_id, str)  # type: ignore[union-attr]

    def test_batch_id_length_is_eight(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        _fill(batcher, 2)
        batch = batcher.try_form_batch()
        assert len(batch.batch_id) == 8  # type: ignore[union-attr]

    def test_batch_created_at_is_float(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        _fill(batcher, 2)
        batch = batcher.try_form_batch()
        assert isinstance(batch.created_at, float)  # type: ignore[union-attr]

    def test_batch_created_at_is_recent(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        _fill(batcher, 2)
        before = time.monotonic()
        batch = batcher.try_form_batch()
        after = time.monotonic()
        assert before <= batch.created_at <= after  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# try_form_batch – max_wait_ms trigger
# ---------------------------------------------------------------------------


class TestTryFormBatchTimeout:
    def test_returns_batch_when_oldest_request_old_enough(self):
        batcher = RequestBatcher(max_batch_size=10, max_wait_ms=0.0)
        batcher.enqueue(_req())
        # With max_wait_ms=0 the very first call should trigger
        batch = batcher.try_form_batch()
        assert batch is not None

    def test_batch_requests_removed_from_queue_after_timeout(self):
        batcher = RequestBatcher(max_batch_size=10, max_wait_ms=0.0)
        batcher.enqueue(_req())
        batcher.try_form_batch()
        assert batcher.queue_size() == 0


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_highest_priority_selected_first(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        low = BatchRequest(request_id="low", payload={}, priority=0)
        high = BatchRequest(request_id="high", payload={}, priority=99)
        extra = BatchRequest(request_id="mid", payload={}, priority=50)
        batcher.enqueue(low)
        batcher.enqueue(high)
        batcher.enqueue(extra)

        batch = batcher.try_form_batch()
        assert batch is not None
        batched_ids = {r.request_id for r in batch.requests}
        assert "high" in batched_ids
        assert "extra" not in batched_ids or "mid" in batched_ids
        # low-priority item should NOT be in the batch
        assert "low" not in batched_ids

    def test_low_priority_left_in_queue(self):
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=100_000)
        batcher.enqueue(BatchRequest(request_id="low", payload={}, priority=0))
        batcher.enqueue(BatchRequest(request_id="hi1", payload={}, priority=10))
        batcher.enqueue(BatchRequest(request_id="hi2", payload={}, priority=10))
        batcher.try_form_batch()
        # "low" should remain
        assert batcher.queue_size() == 1


# ---------------------------------------------------------------------------
# pending_stats
# ---------------------------------------------------------------------------


class TestPendingStats:
    def test_empty_queue_stats(self):
        batcher = RequestBatcher()
        stats = batcher.pending_stats()
        assert stats["queued"] == 0
        assert stats["oldest_wait_ms"] is None

    def test_queued_count_matches(self):
        batcher = RequestBatcher()
        _fill(batcher, 3)
        assert batcher.pending_stats()["queued"] == 3

    def test_oldest_wait_ms_is_nonnegative(self):
        batcher = RequestBatcher()
        batcher.enqueue(_req())
        stats = batcher.pending_stats()
        assert stats["oldest_wait_ms"] is not None
        assert stats["oldest_wait_ms"] >= 0.0
