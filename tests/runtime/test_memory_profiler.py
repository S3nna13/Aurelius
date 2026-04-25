import torch
import torch.nn as nn
import pytest
import dataclasses

from src.runtime.memory_profiler import MemorySnapshot, MemoryProfiler, MEMORY_PROFILER_REGISTRY


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 4)


def _sample_input() -> torch.Tensor:
    return torch.randn(2, 4)


class TestMemorySnapshot:
    def test_is_frozen(self):
        snap = MemorySnapshot(allocated_mb=0.0, reserved_mb=0.0, peak_mb=0.0, device="cpu")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            snap.allocated_mb = 1.0  # type: ignore[misc]

    def test_fields_accessible(self):
        snap = MemorySnapshot(allocated_mb=1.0, reserved_mb=2.0, peak_mb=3.0, device="cuda:0")
        assert snap.allocated_mb == 1.0
        assert snap.reserved_mb == 2.0
        assert snap.peak_mb == 3.0
        assert snap.device == "cuda:0"


class TestMemoryProfiler:
    def test_default_device_is_cpu(self):
        profiler = MemoryProfiler()
        assert profiler.device == "cpu"

    def test_snapshot_cpu_returns_zeros(self):
        profiler = MemoryProfiler(device="cpu")
        snap = profiler.snapshot()
        assert snap.allocated_mb == 0.0
        assert snap.reserved_mb == 0.0
        assert snap.peak_mb == 0.0

    def test_snapshot_cpu_device_field(self):
        profiler = MemoryProfiler(device="cpu")
        snap = profiler.snapshot()
        assert snap.device == "cpu"

    def test_snapshot_returns_memory_snapshot(self):
        profiler = MemoryProfiler(device="cpu")
        snap = profiler.snapshot()
        assert isinstance(snap, MemorySnapshot)

    def test_estimate_activation_mb_arithmetic(self):
        profiler = MemoryProfiler()
        result = profiler.estimate_activation_mb(batch=2, seq_len=16, d_model=64, n_layers=4)
        expected = (2 * 16 * 64 * 4 * 4) / (1024 ** 2)
        assert abs(result - expected) < 1e-9

    def test_estimate_activation_mb_returns_float(self):
        profiler = MemoryProfiler()
        result = profiler.estimate_activation_mb(1, 1, 1, 1)
        assert isinstance(result, float)

    def test_estimate_activation_mb_scales_linearly(self):
        profiler = MemoryProfiler()
        base = profiler.estimate_activation_mb(1, 8, 32, 2)
        doubled = profiler.estimate_activation_mb(2, 8, 32, 2)
        assert abs(doubled - 2 * base) < 1e-9

    def test_oom_guard_cpu_always_true(self):
        profiler = MemoryProfiler(device="cpu")
        model = _tiny_model()
        result = profiler.oom_guard(model, _sample_input())
        assert result is True

    def test_oom_guard_cpu_with_safety_factor(self):
        profiler = MemoryProfiler(device="cpu")
        model = _tiny_model()
        result = profiler.oom_guard(model, _sample_input(), safety_factor=10.0)
        assert result is True

    def test_reset_peak_cpu_noop(self):
        profiler = MemoryProfiler(device="cpu")
        profiler.reset_peak()


class TestMemoryProfilerRegistry:
    def test_registry_has_default_key(self):
        assert "default" in MEMORY_PROFILER_REGISTRY

    def test_registry_default_is_memory_profiler(self):
        assert MEMORY_PROFILER_REGISTRY["default"] is MemoryProfiler

    def test_registry_instantiable(self):
        cls = MEMORY_PROFILER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, MemoryProfiler)
