"""Tests for event emitter."""

from __future__ import annotations

from src.tools.event_emitter import Event, EventEmitter


class TestEventEmitter:
    def test_on_and_emit(self):
        ee = EventEmitter()
        results = []

        def handler(event: Event):
            results.append(event.data)

        ee.on("test", handler)
        ee.emit(Event("test", {"msg": "hello"}))
        assert results == [{"msg": "hello"}]

    def test_off(self):
        ee = EventEmitter()

        def handler(event):
            pass

        ee.on("e", handler)
        assert ee.listener_count("e") == 1
        ee.off("e", handler)
        assert ee.listener_count("e") == 0

    def test_multiple_listeners(self):
        ee = EventEmitter()
        calls = []
        ee.on("e", lambda _: calls.append(1))
        ee.on("e", lambda _: calls.append(2))
        ee.emit(Event("e"))
        assert calls == [1, 2]

    def test_unregistered_event(self):
        ee = EventEmitter()
        ee.emit(Event("unregistered"))  # should not raise
