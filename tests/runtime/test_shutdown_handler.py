"""Tests for shutdown handler."""
from __future__ import annotations

import pytest

from src.runtime.shutdown_handler import ShutdownHandler


class TestShutdownHandler:
    def test_register_and_shutdown(self):
        sh = ShutdownHandler()
        called = False
        def handler():
            nonlocal called
            called = True
        sh.register(handler)
        sh.shutdown()
        assert called is True

    def test_multiple_handlers(self):
        sh = ShutdownHandler()
        calls = []
        sh.register(lambda: calls.append(1))
        sh.register(lambda: calls.append(2))
        sh.shutdown()
        assert calls == [1, 2]

    def test_idempotent(self):
        sh = ShutdownHandler()
        count = 0
        def inc():
            nonlocal count
            count += 1
        sh.register(inc)
        sh.shutdown()
        sh.shutdown()
        assert count == 1

    def test_handler_exception_does_not_block(self):
        sh = ShutdownHandler()
        results = []
        def fails():
            raise ValueError("boom")
        def succeeds():
            results.append("ok")
        sh.register(fails)
        sh.register(succeeds)
        sh.shutdown()
        assert results == ["ok"]