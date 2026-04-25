"""Tests for src/profiling/cpu_profiler.py"""
from __future__ import annotations

import pytest

from src.profiling.cpu_profiler import (
    CPU_PROFILER_REGISTRY,
    CPUProfiler,
    CPUSample,
)


@pytest.fixture
def profiler():
    return CPUProfiler(max_samples=100)


class TestCPUSample:
    def test_fields(self):
        s = CPUSample(timestamp=1.0, core_pcts=[10.0, 20.0, 30.0])
        assert len(s.core_pcts) == 3


class TestCPUProfilerRecord:
    def test_record_accepts_none(self, profiler):
        s = profiler.record()
        assert isinstance(s, CPUSample)
        assert len(s.core_pcts) > 0

    def test_record_accepts_explicit(self, profiler):
        pcts = [50.0, 60.0, 70.0]
        s = profiler.record(core_pcts=pcts)
        assert s.core_pcts == pcts

    def test_record_stored(self, profiler):
        profiler.record(core_pcts=[10.0, 20.0])
        profiler.record(core_pcts=[30.0, 40.0])
        assert len(profiler.samples) == 2


class TestCPUProfilerStats:
    def test_max_core_pct_empty(self, profiler):
        assert profiler.max_core_pct() == 0.0

    def test_max_core_pct(self, profiler):
        profiler.record(core_pcts=[10.0, 90.0])
        profiler.record(core_pcts=[30.0, 70.0])
        assert profiler.max_core_pct() == 90.0

    def test_mean_core_pct_empty(self, profiler):
        assert profiler.mean_core_pct() == 0.0

    def test_mean_core_pct(self, profiler):
        profiler.record(core_pcts=[10.0, 20.0])
        profiler.record(core_pcts=[30.0, 40.0])
        mean = profiler.mean_core_pct()
        assert mean == 25.0


class TestP99Utilization:
    def test_p99_empty(self, profiler):
        assert profiler.p99_utilization() == 0.0

    def test_p99_sorted_property(self, profiler):
        for pcts in [[10.0, 20.0], [80.0, 90.0], [50.0, 60.0]]:
            profiler.record(core_pcts=pcts)
        p99 = profiler.p99_utilization()
        all_vals = sorted(v for s in profiler.samples for v in s.core_pcts)
        idx = min(int(len(all_vals) * 0.99), len(all_vals) - 1)
        assert p99 == all_vals[idx]


class TestMeanUtilization:
    def test_mean_utilization_alias(self, profiler):
        profiler.record(core_pcts=[10.0, 20.0])
        assert profiler.mean_utilization() == profiler.mean_core_pct()


class TestHottestCore:
    def test_hottest_core_empty(self, profiler):
        assert profiler.hottest_core() == -1

    def test_hottest_core(self, profiler):
        profiler.record(core_pcts=[10.0, 90.0, 50.0])
        profiler.record(core_pcts=[20.0, 80.0, 60.0])
        assert profiler.hottest_core() == 1


class TestReset:
    def test_reset_clears_samples(self, profiler):
        profiler.record(core_pcts=[10.0])
        profiler.reset()
        assert len(profiler.samples) == 0


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in CPU_PROFILER_REGISTRY