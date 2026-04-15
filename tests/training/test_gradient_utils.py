"""Tests for src/training/gradient_utils.py.

All tests use tiny nn.Linear models with real backward passes so that actual
gradient tensors are present — no mocking required.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.training.gradient_utils import (
    AdaptiveGradClipper,
    GradientConfig,
    GradientMonitor,
    clip_grad_norm,
    clip_grad_value,
    compute_grad_norm,
    compute_layer_grad_norms,
    detect_gradient_issues,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Linear:
    """2-input → 2-output linear layer (4 weight params + 2 bias params = 6 total)."""
    torch.manual_seed(0)
    return nn.Linear(2, 2)


def _run_backward(model: nn.Module) -> None:
    """Zero grads, do a tiny forward + backward pass."""
    model.zero_grad()
    x = torch.randn(1, 2)
    y = model(x)
    loss = y.sum()
    loss.backward()


def _params(model: nn.Module):
    return list(model.parameters())


# ---------------------------------------------------------------------------
# GradientConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = GradientConfig()
    assert cfg.max_norm == 1.0
    assert cfg.norm_type == 2.0
    assert cfg.clip_value is None
    assert cfg.warn_threshold == 10.0


def test_config_custom():
    cfg = GradientConfig(max_norm=0.5, norm_type=1.0, clip_value=0.1, warn_threshold=5.0)
    assert cfg.max_norm == 0.5
    assert cfg.norm_type == 1.0
    assert cfg.clip_value == 0.1
    assert cfg.warn_threshold == 5.0


# ---------------------------------------------------------------------------
# compute_grad_norm
# ---------------------------------------------------------------------------


def test_compute_grad_norm_returns_scalar():
    model = _tiny_model()
    _run_backward(model)
    norm = compute_grad_norm(_params(model))
    assert isinstance(norm, float)


def test_compute_grad_norm_positive_after_backward():
    model = _tiny_model()
    _run_backward(model)
    norm = compute_grad_norm(_params(model))
    assert norm > 0.0


def test_compute_grad_norm_zero_no_grads():
    model = _tiny_model()
    model.zero_grad()  # ensure no grads
    norm = compute_grad_norm(_params(model))
    assert norm == 0.0


def test_compute_grad_norm_accepts_module():
    """compute_grad_norm should accept an nn.Module directly."""
    model = _tiny_model()
    _run_backward(model)
    norm_module = compute_grad_norm(model)
    norm_params = compute_grad_norm(_params(model))
    assert abs(norm_module - norm_params) < 1e-6


# ---------------------------------------------------------------------------
# clip_grad_norm
# ---------------------------------------------------------------------------


def test_clip_grad_norm_returns_pre_clip_norm():
    model = _tiny_model()
    _run_backward(model)
    pre = compute_grad_norm(_params(model))
    returned = clip_grad_norm(_params(model), max_norm=1e9)  # effectively no clip
    assert abs(returned - pre) < 1e-6


def test_clip_grad_norm_after_clip_norm_le_max_norm():
    model = _tiny_model()
    _run_backward(model)
    max_norm = 0.01  # very small to force clipping
    clip_grad_norm(_params(model), max_norm=max_norm)
    post = compute_grad_norm(_params(model))
    assert post <= max_norm + 1e-6


# ---------------------------------------------------------------------------
# clip_grad_value
# ---------------------------------------------------------------------------


def test_clip_grad_value_clips_to_range():
    model = _tiny_model()
    _run_backward(model)
    clip_value = 1e-4
    clip_grad_value(_params(model), clip_value=clip_value)
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.abs().max().item() <= clip_value + 1e-9


# ---------------------------------------------------------------------------
# GradientMonitor
# ---------------------------------------------------------------------------


def test_gradient_monitor_record_returns_correct_keys():
    model = _tiny_model()
    _run_backward(model)
    cfg = GradientConfig(max_norm=1.0)
    monitor = GradientMonitor(cfg)
    result = monitor.record(_params(model))
    assert set(result.keys()) == {"grad_norm", "clipped_norm", "was_clipped", "clip_ratio"}


def test_gradient_monitor_was_clipped_true_when_norm_exceeds_max():
    model = _tiny_model()
    cfg = GradientConfig(max_norm=1e-10)  # essentially zero → clipping always triggers
    monitor = GradientMonitor(cfg)
    _run_backward(model)
    result = monitor.record(_params(model))
    assert result["was_clipped"] == 1.0


def test_gradient_monitor_was_clipped_false_when_norm_small():
    model = _tiny_model()
    cfg = GradientConfig(max_norm=1e10)  # effectively infinite → no clip
    monitor = GradientMonitor(cfg)
    _run_backward(model)
    result = monitor.record(_params(model))
    assert result["was_clipped"] == 0.0


def test_gradient_monitor_get_stats_keys_present():
    model = _tiny_model()
    cfg = GradientConfig(max_norm=1.0)
    monitor = GradientMonitor(cfg)
    for _ in range(5):
        _run_backward(model)
        monitor.record(_params(model))
    stats = monitor.get_stats()
    assert set(stats.keys()) == {"mean_grad_norm", "max_grad_norm", "clip_frequency", "n_steps"}


def test_gradient_monitor_clip_frequency_in_0_1():
    model = _tiny_model()
    cfg = GradientConfig(max_norm=1.0)
    monitor = GradientMonitor(cfg)
    for _ in range(10):
        _run_backward(model)
        monitor.record(_params(model))
    stats = monitor.get_stats()
    assert 0.0 <= stats["clip_frequency"] <= 1.0


def test_gradient_monitor_reset_clears_history():
    model = _tiny_model()
    cfg = GradientConfig(max_norm=1.0)
    monitor = GradientMonitor(cfg)
    for _ in range(5):
        _run_backward(model)
        monitor.record(_params(model))
    assert len(monitor.get_history()) == 5
    monitor.reset()
    assert len(monitor.get_history()) == 0
    stats = monitor.get_stats()
    assert stats["n_steps"] == 0.0


# ---------------------------------------------------------------------------
# detect_gradient_issues
# ---------------------------------------------------------------------------


def test_detect_gradient_issues_returns_correct_keys():
    model = _tiny_model()
    _run_backward(model)
    result = detect_gradient_issues(_params(model))
    assert set(result.keys()) == {"has_nan", "has_inf", "has_zero_grad", "any_issue"}


def test_detect_gradient_issues_no_issues_after_normal_backward():
    model = _tiny_model()
    _run_backward(model)
    result = detect_gradient_issues(_params(model))
    assert not result["has_nan"]
    assert not result["has_inf"]
    assert not result["any_issue"]


def test_detect_gradient_issues_has_nan():
    model = _tiny_model()
    _run_backward(model)
    # Inject NaN into one gradient
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad[0] = float("nan")
                break
    result = detect_gradient_issues(_params(model))
    assert result["has_nan"]
    assert result["any_issue"]


def test_detect_gradient_issues_has_zero_grad():
    model = _tiny_model()
    model.zero_grad()  # gradients are None
    result = detect_gradient_issues(_params(model))
    assert result["has_zero_grad"]
    assert result["any_issue"]


# ---------------------------------------------------------------------------
# compute_layer_grad_norms
# ---------------------------------------------------------------------------


def test_compute_layer_grad_norms_dict_has_param_names():
    model = _tiny_model()
    _run_backward(model)
    layer_norms = compute_layer_grad_norms(model)
    assert isinstance(layer_norms, dict)
    assert len(layer_norms) > 0
    named_params = {name for name, p in model.named_parameters() if p.grad is not None}
    assert set(layer_norms.keys()) == named_params


def test_compute_layer_grad_norms_values_positive():
    model = _tiny_model()
    _run_backward(model)
    layer_norms = compute_layer_grad_norms(model)
    for name, norm in layer_norms.items():
        assert isinstance(norm, float)
        assert norm >= 0.0


# ---------------------------------------------------------------------------
# AdaptiveGradClipper
# ---------------------------------------------------------------------------


def test_adaptive_grad_clipper_threshold_grows_with_data():
    clipper = AdaptiveGradClipper(percentile=95.0, window=100)
    # Initially no threshold
    assert clipper.get_clip_threshold() == 0.0
    # Feed monotonically increasing norms
    for i in range(1, 11):
        clipper.update(float(i))
    threshold_small = clipper.get_clip_threshold()
    for i in range(11, 21):
        clipper.update(float(i))
    threshold_large = clipper.get_clip_threshold()
    assert threshold_large > threshold_small


def test_adaptive_grad_clipper_clip_returns_pre_clip_norm():
    model = _tiny_model()
    _run_backward(model)
    clipper = AdaptiveGradClipper(percentile=50.0, window=50)
    # Warm up with a small threshold
    for _ in range(10):
        clipper.update(1e-8)
    pre = clipper.clip(_params(model))
    assert isinstance(pre, float)
    assert pre >= 0.0


def test_adaptive_grad_clipper_window_respects_maxlen():
    clipper = AdaptiveGradClipper(percentile=50.0, window=5)
    for i in range(20):
        clipper.update(float(i))
    # Only last 5 values should matter: [15,16,17,18,19]
    threshold = clipper.get_clip_threshold()
    # 50th percentile of [15,16,17,18,19] should be 17.0
    assert abs(threshold - 17.0) < 1e-6
