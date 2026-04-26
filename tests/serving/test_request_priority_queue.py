"""Tests for request_priority_queue.py."""

from __future__ import annotations

import threading

import pytest

from src.serving.request_priority_queue import (
    PRIORITY_QUEUE_REGISTRY,
    RequestPriorityQueue,
)

# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_dequeue_returns_lowest_number_first(self):
        q = RequestPriorityQueue()
        q.enqueue("low", 5, {})
        q.enqueue("high", 0, {})
        q.enqueue("mid", 2, {})
        assert q.dequeue() == ("high", {})
        assert q.dequeue() == ("mid", {})
        assert q.dequeue() == ("low", {})

    def test_list_by_priority_order(self):
        q = RequestPriorityQueue()
        q.enqueue("c", 5, {"k": 3})
        q.enqueue("a", 0, {"k": 1})
        q.enqueue("b", 2, {"k": 2})
        assert q.list_by_priority() == [
            (0, "a", {"k": 1}),
            (2, "b", {"k": 2}),
            (5, "c", {"k": 3}),
        ]

    def test_dequeue_empty_returns_none(self):
        q = RequestPriorityQueue()
        assert q.dequeue() is None


# ---------------------------------------------------------------------------
# FIFO within priority
# ---------------------------------------------------------------------------


class TestFifoWithinPriority:
    def test_same_priority_fifo(self):
        q = RequestPriorityQueue()
        q.enqueue("first", 1, {"order": 1})
        q.enqueue("second", 1, {"order": 2})
        q.enqueue("third", 1, {"order": 3})
        assert q.dequeue() == ("first", {"order": 1})
        assert q.dequeue() == ("second", {"order": 2})
        assert q.dequeue() == ("third", {"order": 3})

    def test_fifo_preserved_in_list_by_priority(self):
        q = RequestPriorityQueue()
        q.enqueue("b", 3, {})
        q.enqueue("a", 3, {})
        q.enqueue("c", 3, {})
        result = q.list_by_priority()
        ids = [rid for _prio, rid, _pld in result]
        assert ids == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# Peek
# ---------------------------------------------------------------------------


class TestPeek:
    def test_peek_returns_highest_without_removing(self):
        q = RequestPriorityQueue()
        q.enqueue("x", 1, {"v": 1})
        assert q.peek() == ("x", {"v": 1})
        assert len(q) == 1
        assert q.dequeue() == ("x", {"v": 1})

    def test_peek_empty_returns_none(self):
        q = RequestPriorityQueue()
        assert q.peek() is None


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing(self):
        q = RequestPriorityQueue()
        q.enqueue("a", 1, {})
        q.enqueue("b", 0, {})
        assert q.remove("a") is True
        assert len(q) == 1
        assert q.dequeue() == ("b", {})
        assert q.dequeue() is None

    def test_remove_nonexistent(self):
        q = RequestPriorityQueue()
        assert q.remove("missing") is False

    def test_remove_after_dequeue(self):
        q = RequestPriorityQueue()
        q.enqueue("a", 1, {})
        q.dequeue()
        assert q.remove("a") is False

    def test_remove_allows_heap_cleanup(self):
        q = RequestPriorityQueue()
        q.enqueue("a", 1, {})
        q.enqueue("b", 0, {})
        q.remove("b")
        # b is still in the heap but should be skipped by dequeue
        assert q.peek() == ("a", {})
        assert q.dequeue() == ("a", {})
        assert q.dequeue() is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_request_id_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue("", 0, {})

    def test_long_request_id_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue("x" * 129, 0, {})

    def test_request_id_exactly_128_ok(self):
        q = RequestPriorityQueue()
        q.enqueue("x" * 128, 0, {})
        assert len(q) == 1

    def test_non_string_request_id_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue(123, 0, {})  # type: ignore[arg-type]

    def test_negative_priority_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue("a", -1, {})

    def test_priority_over_99_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue("a", 100, {})

    def test_non_int_priority_raises(self):
        q = RequestPriorityQueue()
        with pytest.raises(ValueError):
            q.enqueue("a", "high", {})  # type: ignore[arg-type]

    def test_duplicate_request_id_raises(self):
        q = RequestPriorityQueue()
        q.enqueue("dup", 5, {})
        with pytest.raises(ValueError):
            q.enqueue("dup", 5, {})


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


class TestLength:
    def test_len_increments_and_decrements(self):
        q = RequestPriorityQueue()
        assert len(q) == 0
        q.enqueue("a", 0, {})
        assert len(q) == 1
        q.enqueue("b", 0, {})
        assert len(q) == 2
        q.dequeue()
        assert len(q) == 1
        q.dequeue()
        assert len(q) == 0

    def test_len_after_remove(self):
        q = RequestPriorityQueue()
        q.enqueue("a", 0, {})
        q.remove("a")
        assert len(q) == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_enqueue(self):
        q = RequestPriorityQueue()
        num_threads = 10
        items_per_thread = 100
        lock = threading.Lock()
        enqueued: list[str] = []

        def worker(tid: int) -> None:
            for i in range(items_per_thread):
                rid = f"req-{tid}-{i}"
                q.enqueue(rid, priority=i % 100, payload={"i": i})
                with lock:
                    enqueued.append(rid)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(q) == num_threads * items_per_thread
        assert len(enqueued) == num_threads * items_per_thread
        assert len(set(enqueued)) == len(enqueued)

    def test_concurrent_enqueue_and_dequeue(self):
        q = RequestPriorityQueue()
        num_threads = 10
        items_per_thread = 100
        lock = threading.Lock()
        enqueued: list[str] = []
        dequeued: list[str] = []

        def enqueuer(tid: int) -> None:
            for i in range(items_per_thread):
                rid = f"req-{tid}-{i}"
                q.enqueue(rid, priority=i % 100, payload={"i": i})
                with lock:
                    enqueued.append(rid)

        def dequeuer() -> None:
            for _ in range(items_per_thread * 2):
                item = q.dequeue()
                if item is not None:
                    with lock:
                        dequeued.append(item[0])

        enqueue_threads = [threading.Thread(target=enqueuer, args=(i,)) for i in range(num_threads)]
        dequeue_threads = [threading.Thread(target=dequeuer) for _ in range(num_threads)]

        for t in enqueue_threads:
            t.start()
        for t in enqueue_threads:
            t.join()

        for t in dequeue_threads:
            t.start()
        for t in dequeue_threads:
            t.join()

        assert len(dequeued) == num_threads * items_per_thread
        assert set(enqueued) == set(dequeued)
        assert len(set(dequeued)) == len(dequeued)
        assert len(q) == 0

    def test_concurrent_remove(self):
        q = RequestPriorityQueue()
        for i in range(500):
            q.enqueue(f"req-{i}", i % 100, {})

        removed_count = 0
        lock = threading.Lock()

        def remover() -> None:
            nonlocal removed_count
            for i in range(500):
                if q.remove(f"req-{i}"):
                    with lock:
                        removed_count += 1

        threads = [threading.Thread(target=remover) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert removed_count == 500
        assert len(q) == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_queue_present(self):
        assert "default" in PRIORITY_QUEUE_REGISTRY
        assert isinstance(PRIORITY_QUEUE_REGISTRY["default"], RequestPriorityQueue)
