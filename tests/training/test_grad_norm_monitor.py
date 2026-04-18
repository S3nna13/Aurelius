"""Tests for gradient norm monitoring utilities."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.grad_norm_monitor import (
    GradNormMonitor,
    GradNormStats,
    clip_grad_norm_adaptive,
    compute_grad_norms,
)


# ------------------------------------------------------------------------------- #
# Helpers                                                                          #
# ------------------------------------------------------------------------------- #

def make_model_with_grads(scale: float = 1.0) -> nn.Module:
    """Return a tiny Linear model with gradients populated."""
    model = nn.Linear(4, 4, bias=False)
    nn.init.constant_(model.weight, 1.0)
    x = torch.ones(2, 4)
    loss = model(x).sum() * scale
    loss.backward()
    return model


def make_model_no_grads() -> nn.Module:
    """Return a model without any gradients."""
    return nn.Linear(4, 4, bias=False)


# ------------------------------------------------------------------------------- #
# Tests                                                                            #
# ------------------------------------------------------------------------------- #

def test_compute_grad_norms_returns_stats():
    """compute_grad_norms should return a GradNormStats instance."""
    model = make_model_with_grads()
    stats = compute_grad_norms(model)
    assert isinstance(stats, GradNormStats)


def test_layer_norms_dict_has_entries():
    """layer_norms should map parameter names to float norms."""
    model = make_model_with_grads()
    stats = compute_grad_norms(model)
    assert len(stats.layer_norms) > 0
    for key, val in stats.layer_norms.items():
        assert isinstance(key, str)
        assert isinstance(val, float)


def test_global_norm_positive_for_nonzero_grads():
    """global_norm should be > 0 when gradients are non-zero."""
    model = make_model_with_grads(scale=1.0)
    stats = compute_grad_norms(model)
    assert stats.global_norm > 0.0


def test_monitor_update_appends_to_history():
    """update() should append the global norm to the history deque."""
    monitor = GradNormMonitor(window_size=10)
    model = make_model_with_grads()
    stats = compute_grad_norms(model)
    assert len(monitor.history) == 0
    monitor.update(stats)
    assert len(monitor.history) == 1
    assert monitor.history[0] == stats.global_norm


def test_adaptive_clip_value_after_10_steps():
    """adaptive_clip_value should return a finite positive value after history fills."""
    monitor = GradNormMonitor(window_size=100, clip_percentile=95.0)
    for i in range(10):
        model = make_model_with_grads(scale=float(i + 1))
        stats = compute_grad_norms(model)
        monitor.update(stats)
    clip_val = monitor.adaptive_clip_value()
    assert math.isfinite(clip_val)
    assert clip_val > 0.0


def test_should_clip_false_for_small_norm():
    """should_clip should return False when current norm is within normal range."""
    monitor = GradNormMonitor()
    # Fill history with norms around 1.0
    for _ in range(20):
        stats = GradNormStats(global_norm=1.0, layer_norms={}, step=0)
        monitor.update(stats)
    small_stats = GradNormStats(global_norm=1.5, layer_norms={}, step=1)
    assert monitor.should_clip(small_stats, multiplier=2.0) is False


def test_should_clip_true_for_huge_norm():
    """should_clip should return True when current norm far exceeds history."""
    monitor = GradNormMonitor()
    for _ in range(20):
        stats = GradNormStats(global_norm=1.0, layer_norms={}, step=0)
        monitor.update(stats)
    big_stats = GradNormStats(global_norm=1000.0, layer_norms={}, step=1)
    assert monitor.should_clip(big_stats, multiplier=2.0) is True


def test_clip_grad_norm_adaptive_clips_correctly():
    """clip_grad_norm_adaptive should clip and report was_clipped=True for large grads."""
    monitor = GradNormMonitor()
    # Fill with small norms
    for _ in range(20):
        monitor.update(GradNormStats(global_norm=0.1, layer_norms={}, step=0))

    # Make a model with large gradients
    model = make_model_with_grads(scale=1000.0)
    actual_norm, was_clipped = clip_grad_norm_adaptive(model, monitor, multiplier=2.0)
    assert actual_norm > 0.0
    assert was_clipped is True


def test_summary_has_required_keys():
    """summary() should return a dict with mean, std, max, min."""
    monitor = GradNormMonitor()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        monitor.update(GradNormStats(global_norm=v, layer_norms={}, step=0))
    s = monitor.summary()
    assert set(s.keys()) == {"mean", "std", "max", "min"}
    assert s["mean"] == pytest.approx(3.0)
    assert s["max"] == pytest.approx(5.0)
    assert s["min"] == pytest.approx(1.0)


def test_window_size_limits_history_length():
    """History deque should not exceed window_size entries."""
    window = 10
    monitor = GradNormMonitor(window_size=window)
    for i in range(50):
        monitor.update(GradNormStats(global_norm=float(i), layer_norms={}, step=i))
    assert len(monitor.history) == window
    # Should contain the last window entries
    assert list(monitor.history)[-1] == 49.0
