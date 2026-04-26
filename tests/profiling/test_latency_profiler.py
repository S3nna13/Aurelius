"""Tests for latency_profiler — fine-grained latency measurement."""

from __future__ import annotations

import time

from src.profiling.latency_profiler import LatencyProfiler, TimerContext


class TestTimerContext:
    def test_context_manager_records_time(self):
        profiler = LatencyProfiler()
        with TimerContext(profiler, "test_op"):
            time.sleep(0.001)
        stats = profiler.get_stats("test_op")
        assert stats["count"] == 1
        assert stats["total_ms"] > 0

    def test_auto_naming(self):
        profiler = LatencyProfiler()
        with TimerContext(profiler):
            time.sleep(0.001)
        assert profiler.get_stats("block_0") is not None


class TestLatencyProfiler:
    def test_record_manual(self):
        profiler = LatencyProfiler()
        profiler.record("op1", 10.0)
        profiler.record("op1", 20.0)
        stats = profiler.get_stats("op1")
        assert stats["count"] == 2
        assert stats["avg_ms"] == 15.0

    def test_get_stats_nonexistent(self):
        profiler = LatencyProfiler()
        assert profiler.get_stats("nothing") is None

    def test_summary(self):
        profiler = LatencyProfiler()
        profiler.record("a", 5.0)
        profiler.record("b", 15.0)
        summary = profiler.summary()
        assert len(summary) == 2

    def test_reset(self):
        profiler = LatencyProfiler()
        profiler.record("x", 1.0)
        profiler.reset()
        assert profiler.get_stats("x") is None
