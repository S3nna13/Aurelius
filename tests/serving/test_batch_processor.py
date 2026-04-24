"""Tests for batch_processor.py."""

from __future__ import annotations

import uuid

import pytest

from src.serving.batch_processor import (
    BatchProcessor,
    BatchRequest,
    BatchResult,
    QueueFullError,
)


def make_req(priority: int = 5, prompts: list[str] | None = None) -> BatchRequest:
    return BatchRequest(
        request_id=str(uuid.uuid4()),
        prompts=prompts or ["hello world"],
        priority=priority,
    )


def test_enqueue_returns_request_id():
    p = BatchProcessor()
    req = make_req()
    rid = p.enqueue(req)
    assert rid == req.request_id


def test_queue_depth_increments():
    p = BatchProcessor()
    p.enqueue(make_req())
    p.enqueue(make_req())
    assert p.queue_depth() == 2


def test_queue_full_raises():
    p = BatchProcessor(max_queue_size=2)
    p.enqueue(make_req())
    p.enqueue(make_req())
    with pytest.raises(QueueFullError):
        p.enqueue(make_req())


def test_dequeue_batch_respects_max_batch_size():
    p = BatchProcessor(max_batch_size=3)
    for _ in range(5):
        p.enqueue(make_req())
    batch = p.dequeue_batch()
    assert len(batch) == 3
    assert p.queue_depth() == 2


def test_dequeue_empty_queue_returns_empty():
    p = BatchProcessor()
    assert p.dequeue_batch() == []


def test_priority_ordering_lower_first():
    p = BatchProcessor()
    req_low = make_req(priority=8)
    req_high = make_req(priority=1)
    req_mid = make_req(priority=5)
    p.enqueue(req_low)
    p.enqueue(req_high)
    p.enqueue(req_mid)
    batch = p.dequeue_batch()
    assert batch[0].request_id == req_high.request_id
    assert batch[1].request_id == req_mid.request_id
    assert batch[2].request_id == req_low.request_id


def test_process_batch_returns_results():
    p = BatchProcessor()
    req = make_req(prompts=["hello", "world"])
    results = p.process_batch([req])
    assert len(results) == 1
    assert isinstance(results[0], BatchResult)


def test_process_batch_one_output_per_prompt():
    p = BatchProcessor()
    req = make_req(prompts=["a", "b", "c"])
    result = p.process_batch([req])[0]
    assert len(result.outputs) == 3
    assert len(result.token_counts) == 3


def test_process_batch_stub_output_format():
    p = BatchProcessor()
    req = make_req(prompts=["hello world prompt"])
    result = p.process_batch([req])[0]
    assert result.outputs[0].startswith("output_0 for ")


def test_process_batch_latency_scales_with_batch():
    p = BatchProcessor()
    reqs = [make_req() for _ in range(4)]
    results = p.process_batch(reqs)
    for r in results:
        assert r.latency_ms == pytest.approx(40.0)


def test_run_once_processes_queued_requests():
    p = BatchProcessor()
    p.enqueue(make_req())
    p.enqueue(make_req())
    results = p.run_once()
    assert len(results) == 2
    assert p.queue_depth() == 0


def test_run_once_empty_queue_returns_empty():
    p = BatchProcessor()
    assert p.run_once() == []


def test_stats_keys():
    p = BatchProcessor(max_batch_size=16, max_queue_size=128)
    s = p.stats()
    assert "queued" in s
    assert "max_queue_size" in s
    assert "max_batch_size" in s


def test_stats_values_reflect_state():
    p = BatchProcessor(max_batch_size=16, max_queue_size=128)
    p.enqueue(make_req())
    s = p.stats()
    assert s["queued"] == 1
    assert s["max_batch_size"] == 16
    assert s["max_queue_size"] == 128
