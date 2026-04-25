"""Tests for src/profiling/power_profiler.py"""
from __future__ import annotations

import pytest

from src.profiling.power_profiler import (
    POWER_PROFILER_REGISTRY,
    PowerProfiler,
    PowerReading,
    TDPConfig,
)


@pytest.fixture
def profiler():
    return PowerProfiler(max_readings=100)


class TestPowerReading:
    def test_fields(self):
        r = PowerReading(timestamp=1.0, power_w=150.5, component="gpu")
        assert r.power_w == 150.5
        assert r.component == "gpu"


class TestPowerProfilerRecord:
    def test_record_stores_reading(self, profiler):
        r = profiler.record(200.0, "gpu")
        assert isinstance(r, PowerReading)
        assert r.power_w == 200.0
        assert r.component == "gpu"
        assert len(profiler) == 1


class TestPowerProfilerStats:
    def test_total_energy_j_empty(self, profiler):
        assert profiler.total_energy_j() == 0.0

    def test_total_energy_j_single_reading(self, profiler):
        profiler.record(100.0)
        assert profiler.total_energy_j() == 0.0

    def test_total_energy_j_with_readings(self, profiler):
        profiler._readings.clear()
        profiler._readings.append(PowerReading(timestamp=0.0, power_w=100.0))
        profiler._readings.append(PowerReading(timestamp=1.0, power_w=200.0))
        energy = profiler.total_energy_j()
        assert energy > 0.0

    def test_mean_power_w_empty(self, profiler):
        assert profiler.mean_power_w() == 0.0

    def test_mean_power_w(self, profiler):
        profiler.record(100.0)
        profiler.record(200.0)
        mean = profiler.mean_power_w()
        assert mean == 150.0

    def test_peak_power_w_empty(self, profiler):
        assert profiler.peak_power_w() == 0.0

    def test_peak_power_w(self, profiler):
        profiler.record(50.0)
        profiler.record(300.0)
        profiler.record(100.0)
        assert profiler.peak_power_w() == 300.0


class TestEfficiencyScore:
    def test_efficiency_no_readings(self, profiler):
        assert profiler.efficiency_score() == 1.0

    def test_efficiency_score_clamped_to_one(self, profiler):
        profiler._readings.clear()
        profiler._readings.append(PowerReading(timestamp=0.0, power_w=100.0))
        profiler._readings.append(PowerReading(timestamp=1.0, power_w=100.0))
        tdpc = TDPConfig(tdp_w=50.0)
        score = profiler.efficiency_score(tdpc)
        assert score == 1.0

    def test_efficiency_score_below_tdp(self, profiler):
        profiler._readings.clear()
        profiler._readings.append(PowerReading(timestamp=0.0, power_w=50.0))
        profiler._readings.append(PowerReading(timestamp=1.0, power_w=50.0))
        tdpc = TDPConfig(tdp_w=400.0)
        score = profiler.efficiency_score(tdpc)
        assert 0.0 < score < 1.0


class TestReadingsFor:
    def test_readings_for_component(self, profiler):
        profiler.record(100.0, "gpu")
        profiler.record(200.0, "cpu")
        profiler.record(150.0, "gpu")
        gpu_readings = profiler.readings_for("gpu")
        assert len(gpu_readings) == 2
        assert all(r.component == "gpu" for r in gpu_readings)

    def test_readings_for_nonexistent(self, profiler):
        profiler.record(100.0, "gpu")
        cpu_readings = profiler.readings_for("cpu")
        assert cpu_readings == []


class TestFilter:
    def test_filter_none_returns_all(self, profiler):
        profiler.record(100.0, "gpu")
        profiler.record(200.0, "cpu")
        filtered = profiler._filter(None)
        assert len(filtered) == 2

    def test_filter_by_component(self, profiler):
        profiler.record(100.0, "gpu")
        profiler.record(200.0, "cpu")
        filtered = profiler._filter("gpu")
        assert len(filtered) == 1
        assert filtered[0].component == "gpu"


class TestCumulativeEnergy:
    def test_energy_accumulates(self, profiler):
        profiler._readings.clear()
        profiler._readings.append(PowerReading(timestamp=0.0, power_w=100.0))
        profiler._readings.append(PowerReading(timestamp=1.0, power_w=200.0))
        profiler._readings.append(PowerReading(timestamp=2.0, power_w=150.0))
        e1 = profiler.total_energy_j()
        profiler._readings.append(PowerReading(timestamp=3.0, power_w=250.0))
        e2 = profiler.total_energy_j()
        assert e2 > e1


class TestCircularBuffer:
    def test_max_readings_eviction(self):
        pp = PowerProfiler(max_readings=5)
        for i in range(10):
            pp.record(float(i))
        assert len(pp) == 5
        assert pp._readings[-1].power_w == 9.0


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in POWER_PROFILER_REGISTRY