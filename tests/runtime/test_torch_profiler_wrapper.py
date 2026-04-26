"""Tests for src/runtime/torch_profiler_wrapper.py — 8+ tests, CPU-only."""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn

from src.runtime.torch_profiler_wrapper import (
    RUNTIME_REGISTRY,
    AureliusProfiler,
    ProfilerConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_cfg(tmp_path) -> ProfilerConfig:
    return ProfilerConfig(
        enabled=True,
        record_shapes=True,
        with_flops=True,
        export_path=str(tmp_path / "profile.json"),
    )


@pytest.fixture()
def disabled_cfg(tmp_path) -> ProfilerConfig:
    return ProfilerConfig(enabled=False, export_path=str(tmp_path / "disabled.json"))


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU())


# ---------------------------------------------------------------------------
# ProfilerConfig tests
# ---------------------------------------------------------------------------


class TestProfilerConfig:
    def test_defaults(self):
        cfg = ProfilerConfig()
        assert cfg.enabled is True
        assert cfg.record_shapes is True
        assert cfg.with_flops is True
        assert cfg.export_path == "profile.json"

    def test_custom(self, tmp_path):
        path = str(tmp_path / "out.json")
        cfg = ProfilerConfig(enabled=False, record_shapes=False, with_flops=False, export_path=path)
        assert cfg.enabled is False
        assert cfg.record_shapes is False
        assert cfg.with_flops is False
        assert cfg.export_path == path


# ---------------------------------------------------------------------------
# AureliusProfiler tests
# ---------------------------------------------------------------------------


class TestAureliusProfiler:
    def test_context_manager_runs(self, default_cfg, simple_model):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(default_cfg)
        with profiler:
            _ = simple_model(x)
        # No exception → pass

    def test_summarize_before_run_returns_dict(self):
        profiler = AureliusProfiler()
        summary = profiler.summarize()
        assert isinstance(summary, dict)
        assert "total_flops" in summary
        assert "memory_peak_mb" in summary
        assert "top5_ops" in summary

    def test_summarize_after_run_returns_dict(self, default_cfg, simple_model):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(default_cfg)
        with profiler:
            _ = simple_model(x)
        summary = profiler.summarize()
        assert isinstance(summary, dict)
        assert "total_flops" in summary
        assert isinstance(summary["top5_ops"], list)

    def test_summarize_top5_ops_structure(self, default_cfg, simple_model):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(default_cfg)
        with profiler:
            _ = simple_model(x)
        summary = profiler.summarize()
        for op in summary["top5_ops"]:
            assert "name" in op
            assert "cuda_time_us" in op
            assert "count" in op

    def test_disabled_profiler_no_error(self, disabled_cfg, simple_model):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(disabled_cfg)
        with profiler:
            _ = simple_model(x)
        summary = profiler.summarize()
        assert summary["total_flops"] == 0

    def test_export_chrome_trace_creates_file(self, default_cfg, simple_model, tmp_path):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(default_cfg)
        with profiler:
            _ = simple_model(x)
        trace_path = str(tmp_path / "trace.json")
        profiler.export_chrome_trace(trace_path)
        assert os.path.exists(trace_path)

    def test_export_chrome_trace_valid_json(self, default_cfg, simple_model, tmp_path):
        x = torch.randn(2, 8)
        profiler = AureliusProfiler(default_cfg)
        with profiler:
            _ = simple_model(x)
        trace_path = str(tmp_path / "trace2.json")
        profiler.export_chrome_trace(trace_path)
        with open(trace_path) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_registry_entry(self):
        assert "profiler" in RUNTIME_REGISTRY
        assert RUNTIME_REGISTRY["profiler"] is AureliusProfiler
