"""Tests for src/profiling/gpu_profiler.py"""
from __future__ import annotations

import pytest

from src.profiling.gpu_profiler import (
    GPU_PROFILER_REGISTRY,
    GPUProfiler,
    GPUStats,
)


@pytest.fixture
def profiler():
    return GPUProfiler(max_history=100)


class TestGPUStats:
    def test_fields(self):
        s = GPUStats(
            timestamp=1.0,
            utilization_pct=50.0,
            memory_used_mb=4096.0,
            memory_total_mb=16384.0,
        )
        assert s.utilization_pct == 50.0
        assert s.memory_used_mb == 4096.0


class TestGPUProfilerRecord:
    def test_record_stores(self, profiler):
        s = profiler.record(utilization_pct=75.0, memory_used_mb=8192.0, memory_total_mb=16384.0)
        assert len(profiler.history) == 1
        assert s.utilization_pct == 75.0

    def test_record_generates_defaults(self, profiler):
        s = profiler.record()
        assert 0.0 <= s.utilization_pct <= 100.0
        assert s.memory_total_mb > 0


class TestGPUProfilerLatest:
    def test_latest_returns_last(self, profiler):
        profiler.record(utilization_pct=50.0)
        profiler.record(utilization_pct=80.0)
        latest = profiler.latest()
        assert latest is not None
        assert latest.utilization_pct == 80.0

    def test_latest_empty_returns_none(self, profiler):
        assert profiler.latest() is None


class TestGPUProfilerHistory:
    def test_history_returns_all(self, profiler):
        profiler.record(utilization_pct=30.0)
        profiler.record(utilization_pct=60.0)
        assert len(profiler.history) == 2


class TestPeakStats:
    def test_peak_utilization_empty(self, profiler):
        assert profiler.peak_utilization() == 0.0

    def test_peak_utilization(self, profiler):
        profiler.record(utilization_pct=30.0)
        profiler.record(utilization_pct=90.0)
        profiler.record(utilization_pct=50.0)
        assert profiler.peak_utilization() == 90.0

    def test_peak_memory_mb_empty(self, profiler):
        assert profiler.peak_memory_mb() == 0.0

    def test_peak_memory_mb(self, profiler):
        profiler.record(memory_used_mb=2048.0)
        profiler.record(memory_used_mb=8192.0)
        profiler.record(memory_used_mb=4096.0)
        assert profiler.peak_memory_mb() == 8192.0


class TestSummary:
    def test_summary_keys(self, profiler):
        profiler.record(utilization_pct=50.0)
        s = profiler.summary()
        assert "n_samples" in s
        assert "peak_util_pct" in s
        assert "peak_mem_mb" in s
        assert "mean_util_pct" in s

    def test_summary_empty(self, profiler):
        s = profiler.summary()
        assert s["n_samples"] == 0


class TestMemoryStats:
    def test_memory_free_mb(self, profiler):
        profiler.record(memory_used_mb=4096.0, memory_total_mb=16384.0)
        assert profiler.memory_free_mb() == 12288.0

    def test_memory_free_mb_empty(self, profiler):
        assert profiler.memory_free_mb() == 0.0

    def test_memory_utilization_pct(self, profiler):
        profiler.record(memory_used_mb=8192.0, memory_total_mb=16384.0)
        util = profiler.memory_utilization_pct()
        assert util == 50.0

    def test_memory_utilization_pct_empty(self, profiler):
        assert profiler.memory_utilization_pct() == 0.0


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in GPU_PROFILER_REGISTRY