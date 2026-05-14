"""Tests for EventBus and Event."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.observability.event_bus import Event, EventBus


class TestEvent:
    def test_event_fields(self) -> None:
        ev = Event(event_type="tool.call", payload={"name": "calc"}, trace_id="t1")
        assert ev.event_type == "tool.call"
        assert ev.payload == {"name": "calc"}
        assert ev.trace_id == "t1"
        assert isinstance(ev.timestamp, float)

    def test_event_immutable(self) -> None:
        ev = Event(event_type="x")
        with pytest.raises(AttributeError):
            ev.event_type = "y"  # type: ignore[misc]


class TestEventBus:
    def test_publish_delivers(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        def handler(ev: Event) -> None:
            received.append(ev)

        bus.subscribe("foo", handler)
        ev = Event(event_type="foo", payload={"x": 1})
        delivered = bus.publish(ev)
        assert delivered == 1
        assert len(received) == 1
        assert received[0].payload["x"] == 1

    def test_publish_typed(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("bar", lambda ev: received.append(ev), weak=False)
        ev = bus.publish_typed("bar", {"y": 2}, trace_id="t2")
        assert ev.event_type == "bar"
        import gc

        gc.collect()
        assert ev.trace_id == "t2"
        assert len(received) == 1

    def test_unsubscribe(self) -> None:
        bus = EventBus()

        def handler(ev: Event) -> None:
            pass

        bus.subscribe("baz", handler)
        assert bus.subscriber_count("baz") == 1
        removed = bus.unsubscribe("baz", handler)
        assert removed is True
        assert bus.subscriber_count("baz") == 0

    def test_unsubscribe_missing(self) -> None:
        bus = EventBus()

        def handler(ev: Event) -> None:
            pass

        assert bus.unsubscribe("baz", handler) is False

    def test_no_subscribers(self) -> None:
        bus = EventBus()
        ev = Event(event_type="lonely")
        assert bus.publish(ev) == 0

    def test_handler_exception_ignored(self) -> None:
        bus = EventBus()

        def bad(ev: Event) -> None:
            raise RuntimeError("boom")

        def good(ev: Event) -> None:
            received.append(ev)

        received: list[Event] = []
        bus.subscribe("x", bad)
        bus.subscribe("x", good)
        assert bus.publish(Event(event_type="x")) == 1
        assert len(received) == 1

    def test_history(self) -> None:
        bus = EventBus()
        bus.publish_typed("a", {"i": 1})
        bus.publish_typed("b", {"i": 2})
        bus.publish_typed("a", {"i": 3})
        hist = bus.get_history(event_type="a")
        assert len(hist) == 2
        assert hist[0].payload["i"] == 3

    def test_history_limit(self) -> None:
        bus = EventBus()
        for i in range(5):
            bus.publish_typed("x", {"i": i})
        assert len(bus.get_history(limit=2)) == 2

    def test_history_trace_filter(self) -> None:
        bus = EventBus()
        bus.publish_typed("x", {}, trace_id="t1")
        bus.publish_typed("x", {}, trace_id="t2")
        assert len(bus.get_history(trace_id="t2")) == 1

    def test_clear_history(self) -> None:
        bus = EventBus()
        bus.publish_typed("x")
        bus.clear_history()
        assert len(bus.get_history()) == 0

    def test_history_limit_eviction(self) -> None:
        bus = EventBus()
        # monkey-patch history limit for test speed
        bus._history_limit = 3
        for i in range(5):
            bus.publish_typed("x", {"i": i})
        hist = bus.get_history()
        assert len(hist) == 3
        assert hist[0].payload["i"] == 4

    def test_weak_ref_cleanup(self) -> None:
        bus = EventBus()

        def make_handler() -> Any:
            captured: list[Event] = []

            def handler(ev: Event) -> None:
                captured.append(ev)

            bus.subscribe("tmp", handler, weak=True)
            return captured

        make_handler()
        import gc

        gc.collect()
        bus.publish(Event(event_type="tmp"))
        # handler should have been collected; subscriber count drops after publish triggers pruning
        assert bus.subscriber_count("tmp") == 0

    def test_strong_ref_kept(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        def handler(ev: Event) -> None:
            received.append(ev)

        bus.subscribe("keep", handler, weak=False)
        import gc

        gc.collect()
        bus.publish_typed("keep")
        assert len(received) == 1

    def test_thread_safety(self) -> None:
        bus = EventBus()
        lock = threading.Lock()
        count = 0

        def handler(ev: Event) -> None:
            nonlocal count
            with lock:
                count += 1

        bus.subscribe("stress", handler)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    bus.publish_typed("stress")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert count == 1000
