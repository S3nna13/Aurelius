"""Tests for GPU profiler."""
from __future__ import annotations

from src.profiling.gpu_profiler import (
    GPU_PROFILER_REGISTRY,
    GPUProfiler,
    GPUStats,
)


def test_record():
    p = GPUProfiler()
    p.record(utilization_pct=50.0, memory_used_mb=1024.0, memory_total_mb=8192.0)
    latest = p.latest()
    if not isinstance(latest, GPUStats):
        raise ValueError(f"Expected GPUStats, got {type(latest)}")
    if latest.utilization_pct != 50.0:
        raise ValueError(f"Expected 50.0 utilization, got {latest.utilization_pct}")


def test_record_defaults():
    p = GPUProfiler()
    p.record()
    latest = p.latest()
    if latest is None:
        raise ValueError("Expected GPUStats, got None")
    if latest.utilization_pct != 0.0:
        raise ValueError(f"Expected 0.0 utilization, got {latest.utilization_pct}")


def test_latest_none():
    p = GPUProfiler()
    latest = p.latest()
    if latest is not None:
        raise ValueError(f"Expected None for no history, got {latest}")


def test_history():
    p = GPUProfiler()
    p.record(10.0, 256.0, 1024.0)
    p.record(20.0, 512.0, 1024.0)
    h = p.history()
    if len(h) != 2:
        raise ValueError(f"Expected 2 history entries, got {len(h)}")


def test_peak_utilization():
    p = GPUProfiler()
    p.record(30.0, 256.0, 1024.0)
    p.record(80.0, 512.0, 1024.0)
    peak = p.peak_utilization()
    if peak != 80.0:
        raise ValueError(f"Expected peak 80.0, got {peak}")


def test_peak_memory_mb():
    p = GPUProfiler()
    p.record(30.0, 256.0, 1024.0)
    p.record(80.0, 2048.0, 4096.0)
    peak = p.peak_memory_mb()
    if peak != 2048.0:
        raise ValueError(f"Expected peak 2048.0, got {peak}")


def test_summary():
    p = GPUProfiler()
    p.record(50.0, 1024.0, 4096.0)
    s = p.summary()
    if s["device"] != 0:
        raise ValueError(f"Expected device=0, got {s['device']}")
    if s["samples"] != 1:
        raise ValueError(f"Expected samples=1, got {s['samples']}")


def test_memory_free_mb():
    p = GPUProfiler()
    p.record(50.0, 1024.0, 4096.0)
    latest = p.latest()
    free = latest.memory_free_mb()
    if free != 3072.0:
        raise ValueError(f"Expected free 3072.0, got {free}")


def test_memory_utilization_pct():
    p = GPUProfiler()
    p.record(50.0, 2048.0, 4096.0)
    latest = p.latest()
    pct = latest.memory_utilization_pct()
    if pct != 50.0:
        raise ValueError(f"Expected 50.0%, got {pct}")


def test_multi_device():
    p = GPUProfiler(num_devices=2)
    p.record(10.0, 256.0, 1024.0, device_id=0)
    p.record(80.0, 512.0, 1024.0, device_id=1)
    latest0 = p.latest(0)
    latest1 = p.latest(1)
    if latest0.utilization_pct != 10.0:
        raise ValueError(f"Expected 10.0 for device 0, got {latest0.utilization_pct}")
    if latest1.utilization_pct != 80.0:
        raise ValueError(f"Expected 80.0 for device 1, got {latest1.utilization_pct}")


def test_registry():
    if "default" not in GPU_PROFILER_REGISTRY:
        raise ValueError("default not in GPU_PROFILER_REGISTRY")
    inst = GPU_PROFILER_REGISTRY["default"]()
    if not isinstance(inst, GPUProfiler):
        raise ValueError(f"Expected GPUProfiler instance, got {type(inst)}")