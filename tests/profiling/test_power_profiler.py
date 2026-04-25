"""Tests for power profiler."""
from __future__ import annotations

import time

from src.profiling.power_profiler import (
    POWER_PROFILER_REGISTRY,
    PowerProfiler,
    PowerReading,
)


def test_record():
    p = PowerProfiler()
    r = p.record(100.0, "cpu")
    if not isinstance(r, PowerReading):
        raise ValueError(f"Expected PowerReading, got {type(r)}")
    if r.power_w != 100.0:
        raise ValueError(f"Expected power_w=100.0, got {r.power_w}")
    if r.component != "cpu":
        raise ValueError(f"Expected component=cpu, got {r.component}")


def test_total_energy_j_single():
    p = PowerProfiler()
    p.record(100.0)
    e = p.total_energy_j()
    if e != 0.0:
        raise ValueError(f"Expected 0.0 for single reading, got {e}")


def test_total_energy_j_multiple():
    p = PowerProfiler()
    p.record(100.0)
    time.sleep(0.01)
    p.record(200.0)
    e = p.total_energy_j()
    if e <= 0.0:
        raise ValueError(f"Expected positive energy, got {e}")


def test_mean_power_w_no_readings():
    p = PowerProfiler()
    m = p.mean_power_w()
    if m != 0.0:
        raise ValueError(f"Expected 0.0 for no readings, got {m}")


def test_mean_power_w_single():
    p = PowerProfiler()
    p.record(150.0, "total")
    m = p.mean_power_w("total")
    if m != 150.0:
        raise ValueError(f"Expected 150.0, got {m}")


def test_peak_power_w():
    p = PowerProfiler()
    p.record(50.0)
    time.sleep(0.01)
    p.record(200.0)
    p = p.peak_power_w()
    if p != 200.0:
        raise ValueError(f"Expected peak 200.0, got {p}")


def test_efficiency_score_no_readings():
    p = PowerProfiler()
    e = p.efficiency_score()
    if e != 1.0:
        raise ValueError(f"Expected 1.0 for no readings, got {e}")


def test_readings_for():
    p = PowerProfiler()
    p.record(100.0, "cpu")
    p.record(200.0, "gpu")
    p.record(150.0, "cpu")
    r = p.readings_for("cpu")
    if len(r) != 2:
        raise ValueError(f"Expected 2 cpu readings, got {len(r)}")


def test_filter():
    p = PowerProfiler()
    p.record(100.0, "cpu")
    p.record(200.0, "gpu")
    r = p._filter("cpu")
    if len(r) != 1:
        raise ValueError(f"Expected 1 filtered reading, got {len(r)}")


def test_circular_buffer():
    p = PowerProfiler(max_readings=3)
    for i in range(5):
        p.record(float(i * 100))
    if len(p._readings) != 3:
        raise ValueError(f"Expected 3 readings (circular buffer), got {len(p._readings)}")


def test_registry():
    if "default" not in POWER_PROFILER_REGISTRY:
        raise ValueError("default not in POWER_PROFILER_REGISTRY")
    inst = POWER_PROFILER_REGISTRY["default"]()
    if not isinstance(inst, PowerProfiler):
        raise ValueError(f"Expected PowerProfiler instance, got {type(inst)}")