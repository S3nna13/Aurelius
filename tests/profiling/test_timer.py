"""Tests for profiling timer."""
from __future__ import annotations

import time

import pytest

from src.profiling.timer import Timer, timed, get_timer, report_all


class TestTimer:
    def test_start_stop(self):
        t = Timer("test")
        t.start()
        time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed > 0.0

    def test_elapsed_before_stop(self):
        t = Timer("test")
        t.start()
        time.sleep(0.01)
        assert t.elapsed > 0.0

    def test_reset(self):
        t = Timer("test")
        t.start()
        time.sleep(0.01)
        t.stop()
        t.reset()
        assert t.elapsed == 0.0

    def test_timed_decorator(self):
        @timed("test_func")
        def sleeper():
            time.sleep(0.01)
            return 42

        result = sleeper()
        assert result == 42

    def test_get_timer(self):
        t = get_timer("shared")
        assert t.name == "shared"

    def test_report_all(self):
        _ = get_timer("report_test")
        report = report_all()
        assert isinstance(report, dict)