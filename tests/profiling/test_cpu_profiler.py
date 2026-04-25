"""Tests for CPU profiler."""
from __future__ import annotations

from src.profiling.cpu_profiler import (
    CPU_PROFILER_REGISTRY,
    CPUProfiler,
    CPUSample,
)


def test_record():
    p = CPUProfiler()
    s = p.record(80.0, [80.0, 75.0, 90.0])
    if not isinstance(s, CPUSample):
        raise ValueError(f"Expected CPUSample, got {type(s)}")
    if s.overall_pct != 80.0:
        raise ValueError(f"Expected overall_pct=80.0, got {s.overall_pct}")
    if len(s.per_core_pct) != 3:
        raise ValueError(f"Expected 3 cores, got {len(s.per_core_pct)}")


def test_record_no_cores():
    p = CPUProfiler()
    s = p.record(50.0)
    if s.num_cores != 0:
        raise ValueError(f"Expected 0 cores, got {s.num_cores}")


def test_max_core_pct():
    p = CPUProfiler()
    s = p.record(80.0, [30.0, 90.0, 60.0])
    m = s.max_core_pct()
    if m != 90.0:
        raise ValueError(f"Expected max_core_pct=90.0, got {m}")


def test_mean_core_pct():
    p = CPUProfiler()
    s = p.record(70.0, [40.0, 80.0, 60.0])
    m = s.mean_core_pct()
    if m != 60.0:
        raise ValueError(f"Expected mean_core_pct=60.0, got {m}")


def test_p99_utilization_empty():
    p = CPUProfiler()
    v = p.p99_utilization()
    if v != 0.0:
        raise ValueError(f"Expected 0.0 for empty, got {v}")


def test_p99_utilization():
    p = CPUProfiler()
    for i in range(100):
        p.record(float(i))
    v = p.p99_utilization()
    if v <= 0.0:
        raise ValueError(f"Expected positive p99, got {v}")


def test_mean_utilization():
    p = CPUProfiler()
    p.record(50.0)
    p.record(100.0)
    m = p.mean_utilization()
    if m != 75.0:
        raise ValueError(f"Expected 75.0, got {m}")


def test_hottest_core():
    p = CPUProfiler()
    p.record(80.0, [80.0, 90.0, 70.0])
    p.record(80.0, [85.0, 88.0, 75.0])
    idx = p.hottest_core()
    if idx != 1:
        raise ValueError(f"Expected hottest_core=1, got {idx}")


def test_hottest_core_empty():
    p = CPUProfiler()
    idx = p.hottest_core()
    if idx is not None:
        raise ValueError(f"Expected None for empty, got {idx}")


def test_reset():
    p = CPUProfiler()
    p.record(80.0, [80.0, 90.0])
    p.reset()
    if len(p._samples) != 0:
        raise ValueError(f"Expected empty samples after reset, got {len(p._samples)}")


def test_samples():
    p = CPUProfiler()
    p.record(80.0, [80.0])
    s = p.samples()
    if len(s) != 1:
        raise ValueError(f"Expected 1 sample, got {len(s)}")


def test_registry():
    if "default" not in CPU_PROFILER_REGISTRY:
        raise ValueError("default not in CPU_PROFILER_REGISTRY")
    inst = CPU_PROFILER_REGISTRY["default"]()
    if not isinstance(inst, CPUProfiler):
        raise ValueError(f"Expected CPUProfiler instance, got {type(inst)}")