import time

from src.serving.request_queue import (
    REQUEST_QUEUE_REGISTRY,
    QueuedRequest,
    QueuePriority,
    RequestQueue,
)


def make_req(req_id, priority=QueuePriority.NORMAL, deadline=None):
    return QueuedRequest(request_id=req_id, priority=priority, deadline=deadline)


def test_enqueue_dequeue_priority_order():
    q = RequestQueue()
    q.enqueue(make_req("low", QueuePriority.LOW))
    q.enqueue(make_req("critical", QueuePriority.CRITICAL))
    q.enqueue(make_req("normal", QueuePriority.NORMAL))
    assert q.dequeue().request_id == "critical"
    assert q.dequeue().request_id == "normal"
    assert q.dequeue().request_id == "low"


def test_critical_before_low():
    q = RequestQueue()
    q.enqueue(make_req("low", QueuePriority.LOW))
    q.enqueue(make_req("critical", QueuePriority.CRITICAL))
    first = q.dequeue()
    assert first.priority == QueuePriority.CRITICAL


def test_maxsize_enforcement():
    q = RequestQueue(maxsize=2)
    assert q.enqueue(make_req("a")) is True
    assert q.enqueue(make_req("b")) is True
    assert q.enqueue(make_req("c")) is False
    assert q.size() == 2


def test_drop_expired():
    q = RequestQueue()
    past = time.monotonic() - 1.0
    future = time.monotonic() + 100.0
    q.enqueue(make_req("expired", deadline=past))
    q.enqueue(make_req("valid", deadline=future))
    q.enqueue(make_req("no_deadline"))
    removed = q.drop_expired()
    assert removed == 1
    assert q.size() == 2
    ids = {q.dequeue().request_id, q.dequeue().request_id}
    assert "expired" not in ids


def test_peek_without_removing():
    q = RequestQueue()
    q.enqueue(make_req("a", QueuePriority.HIGH))
    peeked = q.peek()
    assert peeked.request_id == "a"
    assert q.size() == 1


def test_dequeue_empty_returns_none():
    q = RequestQueue()
    assert q.dequeue() is None


def test_peek_empty_returns_none():
    q = RequestQueue()
    assert q.peek() is None


def test_size_and_is_empty():
    q = RequestQueue()
    assert q.is_empty()
    assert q.size() == 0
    q.enqueue(make_req("x"))
    assert not q.is_empty()
    assert q.size() == 1
    q.dequeue()
    assert q.is_empty()


def test_registry_key():
    assert "default" in REQUEST_QUEUE_REGISTRY
    assert REQUEST_QUEUE_REGISTRY["default"] is RequestQueue
