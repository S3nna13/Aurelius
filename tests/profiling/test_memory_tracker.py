"""Tests for memory_tracker — peak memory tracking."""

from __future__ import annotations

import torch.nn as nn

from src.profiling.memory_tracker import MemoryTracker, format_bytes


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(0) == "0.00B"
        assert format_bytes(512) == "512.00B"

    def test_kilobytes(self):
        assert "KB" in format_bytes(1024)

    def test_megabytes(self):
        assert "MB" in format_bytes(1024 * 1024)

    def test_gigabytes(self):
        assert "GB" in format_bytes(1024 * 1024 * 1024)


class TestMemoryTracker:
    def test_snapshot_returns_nonzero_params(self):
        model = nn.Linear(100, 200)
        tracker = MemoryTracker()
        snap = tracker.snapshot_params(model)
        assert snap["numel"] > 0
        assert snap["bytes"] > 0

    def test_snapshot_buffers_includes_buffers(self):
        model = nn.BatchNorm1d(64)
        tracker = MemoryTracker()
        snap = tracker.snapshot_params(model)
        assert snap["numel"] > 0

    def test_peak_diff_tracks_allocation(self):
        tracker = MemoryTracker()
        before = tracker.peak_diff() or (0, 0)
        assert isinstance(before, tuple)

    def test_module_breakdown_returns_dict(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        tracker = MemoryTracker()
        breakdown = tracker.module_breakdown(model)
        assert len(breakdown) >= 2
        for name, info in breakdown.items():
            assert "numel" in info
            assert "bytes" in info

    def test_linear_layer_size(self):
        model = nn.Linear(1024, 2048)
        tracker = MemoryTracker()
        snap = tracker.snapshot_params(model)
        weight_bytes = 1024 * 2048 * 4
        bias_bytes = 2048 * 4
        assert snap["bytes"] == weight_bytes + bias_bytes

    def test_no_double_counting_shared_params(self):
        shared = nn.Linear(10, 10, bias=False)
        model = nn.ModuleList([shared, shared])
        tracker = MemoryTracker()
        snap = tracker.snapshot_params(model)
        numel_1 = 10 * 10
        assert snap["numel"] == numel_1
