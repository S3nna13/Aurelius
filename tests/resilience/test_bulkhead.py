"""Tests for Bulkhead."""

from __future__ import annotations

import threading
import time

import pytest

from src.resilience.bulkhead import Bulkhead, BulkheadFullError


def _slow(duration: float = 0.1) -> None:
    time.sleep(duration)


class TestBulkheadBasics:
    def test_execute_success(self) -> None:
        bh = Bulkhead(max_concurrent=2)
        assert bh.execute(lambda: 7) == 7

    def test_limits_concurrency(self) -> None:
        bh = Bulkhead(max_concurrent=1)
        order: list[str] = []
        lock = threading.Lock()

        def work(name: str) -> str:
            with lock:
                order.append(f"start-{name}")
            time.sleep(0.05)
            with lock:
                order.append(f"end-{name}")
            return name

        t1 = threading.Thread(target=bh.execute, args=(work, "a"))
        t2 = threading.Thread(target=bh.execute, args=(work, "b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # one must finish before the other starts
        assert order.index("end-a") < order.index("start-b") or order.index("end-b") < order.index("start-a")

    def test_queue_full_raises(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=0, queue_timeout=0.01)

        def blocker() -> None:
            time.sleep(0.2)

        t = threading.Thread(target=bh.execute, args=(blocker,))
        t.start()
        time.sleep(0.02)
        with pytest.raises(BulkheadFullError):
            bh.execute(lambda: None)
        t.join()

    def test_queue_wait_timeout(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=1, queue_timeout=0.01)

        def blocker() -> None:
            time.sleep(0.2)

        t = threading.Thread(target=bh.execute, args=(blocker,))
        t.start()
        time.sleep(0.02)
        with pytest.raises(BulkheadFullError, match="timed out"):
            bh.execute(lambda: None)
        t.join()

    def test_active_and_queue_counts(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=2)
        started = threading.Event()

        def blocker() -> None:
            started.set()
            time.sleep(0.2)

        t = threading.Thread(target=bh.execute, args=(blocker,))
        t.start()
        started.wait()
        assert bh.active_count == 1
        t.join()
        assert bh.active_count == 0

    def test_release_wakes_queued(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=2)
        results: list[int] = []

        def work(x: int) -> None:
            time.sleep(0.03)
            results.append(x)

        threads = [threading.Thread(target=bh.execute, args=(work, i)) for i in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert sorted(results) == [0, 1, 2]

    def test_name_in_error(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=0, name="test-bh")

        def blocker() -> None:
            time.sleep(0.2)

        t = threading.Thread(target=bh.execute, args=(blocker,))
        t.start()
        time.sleep(0.02)
        with pytest.raises(BulkheadFullError, match="test-bh"):
            bh.execute(lambda: None)
        t.join()
