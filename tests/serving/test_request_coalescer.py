"""Tests for src/serving/request_coalescer.py."""

from __future__ import annotations

import threading
import time

import pytest

from src.serving.request_coalescer import RequestCoalescer


def test_coalesces_concurrent_identical_requests():
    rc = RequestCoalescer(ttl_s=5.0)
    compute_count = 0
    lock = threading.Lock()

    def compute() -> str:
        nonlocal compute_count
        with lock:
            compute_count += 1
        time.sleep(0.5)
        return "result"

    results: list[str] = []
    errors: list[BaseException] = []

    def worker():
        try:
            results.append(rc.coalesce("key1", compute))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert compute_count == 1
    assert len(results) == 10
    assert all(r == "result" for r in results)
    assert not errors


def test_different_keys_compute_independently():
    rc = RequestCoalescer(ttl_s=5.0)
    calls: set[str] = set()
    lock = threading.Lock()

    def make_compute(key: str):
        def compute() -> str:
            with lock:
                calls.add(key)
            return f"result-{key}"

        return compute

    assert rc.coalesce("a", make_compute("a")) == "result-a"
    assert rc.coalesce("b", make_compute("b")) == "result-b"
    assert calls == {"a", "b"}


def test_exception_propagates_to_all_waiters():
    rc = RequestCoalescer(ttl_s=5.0)

    class MyError(RuntimeError):
        pass

    def compute() -> str:
        raise MyError("boom")

    errors: list[BaseException] = []

    def worker():
        try:
            rc.coalesce("fail", compute)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 5
    assert all(isinstance(e, MyError) for e in errors)


def test_stale_entry_is_replaced():
    rc = RequestCoalescer(ttl_s=0.05)
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return f"v{calls}"

    assert rc.coalesce("k", compute) == "v1"
    time.sleep(0.1)
    assert rc.coalesce("k", compute) == "v2"
    assert calls == 2


def test_stats_reflect_inflight():
    rc = RequestCoalescer(ttl_s=5.0)
    barrier = threading.Barrier(2)

    def compute() -> str:
        barrier.wait()
        return "x"

    def worker():
        rc.coalesce("k", compute)

    t = threading.Thread(target=worker)
    t.start()
    # Give the leader time to register
    time.sleep(0.05)
    stats = rc.stats()
    assert stats["inflight_count"] == 1
    assert stats["total_waiters"] >= 1
    barrier.wait()
    t.join()


def test_clear_wakes_waiters_with_error():
    rc = RequestCoalescer(ttl_s=60.0)
    started = threading.Event()

    def compute() -> str:
        started.set()
        time.sleep(10)
        return "x"

    errors: list[BaseException] = []

    def leader():
        try:
            rc.coalesce("k", compute)
        except BaseException as exc:
            errors.append(exc)

    def waiter():
        try:
            rc.coalesce("k", compute)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=leader)
    t1.start()
    started.wait(timeout=2.0)
    # Start waiter after leader is computing
    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.05)
    rc.clear()
    t2.join(timeout=2.0)
    # Waiter should have been woken by clear
    assert any("cleared" in str(e).lower() for e in errors)
    t1.join(timeout=0.5)  # don't wait full 10s


def test_ttl_must_be_positive():
    with pytest.raises(ValueError, match="ttl_s must be positive"):
        RequestCoalescer(ttl_s=0.0)
