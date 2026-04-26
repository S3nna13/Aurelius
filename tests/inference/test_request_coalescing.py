"""Tests for src/inference/request_coalescing.py"""

from __future__ import annotations

import threading
import time

import pytest

from src.inference.request_coalescing import (
    CoalescingConfig,
    CoalescingSlot,
    RequestCoalescer,
)


class TestCoalescingConfig:
    def test_default_timeout(self):
        cfg = CoalescingConfig()
        assert cfg.timeout == 30.0

    def test_custom_timeout(self):
        cfg = CoalescingConfig(timeout=10.0)
        assert cfg.timeout == 10.0


class TestCoalescingSlot:
    def test_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(CoalescingSlot)

    def test_fields(self):
        slot = CoalescingSlot(prompt="hello", prompt_hash="abc")
        assert slot.prompt == "hello"
        assert slot.prompt_hash == "abc"
        assert slot.result is None
        assert slot.error is None
        assert isinstance(slot.event, threading.Event)


class TestRequestCoalescerConstruction:
    def test_default_config(self):
        rc = RequestCoalescer()
        assert rc.config.timeout == 30.0

    def test_custom_config(self):
        rc = RequestCoalescer(CoalescingConfig(timeout=5.0))
        assert rc.config.timeout == 5.0

    def test_empty_length(self):
        rc = RequestCoalescer()
        assert len(rc) == 0


class TestRequestCoalescerSubmit:
    def test_first_submitter_is_unique(self):
        rc = RequestCoalescer()
        slot = rc.submit("hello world")
        assert slot.prompt == "hello world"
        s = rc.stats()
        assert s["unique"] == 1.0
        assert s["deduplicated"] == 0.0

    def test_duplicate_prompt_is_deduplicated(self):
        rc = RequestCoalescer()
        rc.submit("hello world")
        slot2 = rc.submit("hello world")
        assert slot2 is not None
        s = rc.stats()
        assert s["unique"] == 1.0
        assert s["deduplicated"] == 1.0

    def test_different_prompts_are_unique(self):
        rc = RequestCoalescer()
        rc.submit("hello")
        rc.submit("world")
        s = rc.stats()
        assert s["unique"] == 2.0
        assert s["deduplicated"] == 0.0

    def test_len_returns_pending_count(self):
        rc = RequestCoalescer()
        assert len(rc) == 0
        rc.submit("test")
        assert len(rc) == 1
        rc.submit("test")
        assert len(rc) == 2


class TestRequestCoalescerResolve:
    def test_resolve_delivers_to_all_waiters(self):
        rc = RequestCoalescer()
        slots = [rc.submit("hello") for _ in range(3)]
        count = rc.resolve("hello", "result_data")
        assert count == 3
        for slot in slots:
            assert slot.result == "result_data"
            assert slot.event.is_set()

    def test_resolve_returns_zero_for_unknown(self):
        rc = RequestCoalescer()
        count = rc.resolve("unknown", "data")
        assert count == 0

    def test_wait_returns_resolved_value(self):
        rc = RequestCoalescer()
        slot = rc.submit("hello")
        rc.resolve("hello", "the_answer")
        result = rc.wait(slot, timeout=1.0)
        assert result == "the_answer"


class TestRequestCoalescerReject:
    def test_reject_delivers_error_to_all_waiters(self):
        rc = RequestCoalescer()
        slots = [rc.submit("hello") for _ in range(2)]
        err = RuntimeError("inference failed")
        count = rc.reject("hello", err)
        assert count == 2
        for slot in slots:
            assert slot.error is err
            assert slot.event.is_set()

    def test_wait_raises_rejected_error(self):
        rc = RequestCoalescer()
        slot = rc.submit("hello")
        rc.reject("hello", ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            rc.wait(slot, timeout=1.0)


class TestRequestCoalescerTimeout:
    def test_wait_raises_timeout(self):
        rc = RequestCoalescer(CoalescingConfig(timeout=0.05))
        slot = rc.submit("hello")
        with pytest.raises(TimeoutError):
            rc.wait(slot)

    def test_wait_with_custom_timeout(self):
        rc = RequestCoalescer()
        slot = rc.submit("hello")
        with pytest.raises(TimeoutError):
            rc.wait(slot, timeout=0.01)

    def test_wait_succeeds_before_timeout(self):
        rc = RequestCoalescer()
        slot = rc.submit("hello")

        def delayed_resolve():
            time.sleep(0.05)
            rc.resolve("hello", "data")

        t = threading.Thread(target=delayed_resolve, daemon=True)
        t.start()
        result = rc.wait(slot, timeout=5.0)
        assert result == "data"


class TestRequestCoalescerClear:
    def test_clear_empties_slots_and_resets_stats(self):
        rc = RequestCoalescer()
        rc.submit("a")
        rc.submit("a")
        rc.submit("b")
        rc.clear()
        assert len(rc) == 0
        s = rc.stats()
        assert s["unique"] == 0.0
        assert s["deduplicated"] == 0.0
        assert s["pending"] == 0.0


class TestRequestCoalescerStats:
    def test_savings_ratio(self):
        rc = RequestCoalescer()
        rc.submit("a")
        rc.submit("a")
        rc.submit("a")
        s = rc.stats()
        assert s["savings_ratio"] == pytest.approx(2.0 / 3.0)

    def test_savings_ratio_no_requests(self):
        rc = RequestCoalescer()
        s = rc.stats()
        assert s["savings_ratio"] == 0.0

    def test_pending_reflects_active_slots(self):
        rc = RequestCoalescer()
        rc.submit("x")
        rc.submit("y")
        rc.submit("x")
        s = rc.stats()
        assert s["pending"] == 3.0


class TestRequestCoalescerThreadSafety:
    def test_concurrent_submit_resolve(self):
        rc = RequestCoalescer()
        n_threads = 10
        barrier = threading.Barrier(n_threads)
        slots: list[CoalescingSlot] = []

        def submitter():
            barrier.wait()
            s = rc.submit("shared")
            slots.append(s)

        threads = [threading.Thread(target=submitter, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        rc.resolve("shared", "done")
        assert all(s.event.is_set() for s in slots)
