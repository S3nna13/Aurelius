"""Tests for src/model/norm_tracking.py.

All tests use tiny configs (small D, B, T) and pure PyTorch only.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.norm_tracking import (
    ActivationTracker,
    LayerStatsHook,
    RunningStats,
    TrackingConfig,
    compute_gradient_norm_per_layer,
    compute_tensor_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> nn.Sequential:
    """Two-layer linear model for testing (D=8)."""
    return nn.Sequential(
        nn.Linear(8, 8, bias=False),
        nn.ReLU(),
        nn.Linear(8, 4, bias=False),
    )


def _forward_backward(model: nn.Module, B: int = 2, D: int = 8):
    """Run one forward + backward pass and return the input tensor."""
    x = torch.randn(B, D)
    out = model(x)
    loss = out.sum()
    loss.backward()
    return x


# ---------------------------------------------------------------------------
# TrackingConfig tests
# ---------------------------------------------------------------------------

class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.track_activations is True
        assert cfg.track_gradients is True
        assert math.isclose(cfg.ema_decay, 0.99)
        assert cfg.log_interval == 100
        assert cfg.percentiles == [0.1, 0.5, 0.9]

    def test_custom_values(self):
        cfg = TrackingConfig(track_activations=False, ema_decay=0.9, log_interval=50)
        assert cfg.track_activations is False
        assert math.isclose(cfg.ema_decay, 0.9)
        assert cfg.log_interval == 50


# ---------------------------------------------------------------------------
# RunningStats tests
# ---------------------------------------------------------------------------

class TestRunningStats:
    def test_update_changes_mean(self):
        rs = RunningStats(decay=0.9)
        rs.update(10.0)
        m0 = rs.mean()
        rs.update(20.0)
        assert rs.mean() != m0, "mean should change after update"

    def test_std_non_negative(self):
        rs = RunningStats(decay=0.99)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.std() >= 0.0

    def test_std_zero_after_single_update(self):
        rs = RunningStats()
        rs.update(42.0)
        assert rs.std() == 0.0, "variance should be 0 after a single update"

    def test_reset_clears_state(self):
        rs = RunningStats(decay=0.9)
        for v in [1.0, 5.0, 10.0]:
            rs.update(v)
        rs.reset()
        assert rs.mean() == 0.0
        assert rs.std() == 0.0
        # After reset, first update should behave as first-ever update
        rs.update(7.0)
        assert math.isclose(rs.mean(), 7.0)

    def test_ema_approaches_constant(self):
        """Mean should converge toward a constant stream of values."""
        rs = RunningStats(decay=0.5)
        for _ in range(50):
            rs.update(10.0)
        assert math.isclose(rs.mean(), 10.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# compute_tensor_stats tests
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"mean", "std", "norm", "max_abs", "fraction_zero", "fraction_nan"}


class TestComputeTensorStats:
    def test_has_all_required_keys(self):
        x = torch.randn(4, 8)
        stats = compute_tensor_stats(x)
        assert REQUIRED_KEYS <= stats.keys()

    def test_fraction_nan_zero_for_clean_tensor(self):
        x = torch.randn(4, 8)
        stats = compute_tensor_stats(x)
        assert stats["fraction_nan"] == 0.0

    def test_fraction_zero_correct_for_zero_tensor(self):
        x = torch.zeros(4, 8)
        stats = compute_tensor_stats(x)
        assert math.isclose(stats["fraction_zero"], 1.0)

    def test_fraction_nan_nonzero_for_nan_tensor(self):
        x = torch.full((4, 8), float("nan"))
        stats = compute_tensor_stats(x)
        assert stats["fraction_nan"] == 1.0

    def test_norm_positive_for_nonzero_tensor(self):
        x = torch.ones(4, 8)
        stats = compute_tensor_stats(x)
        assert stats["norm"] > 0.0

    def test_max_abs_correct(self):
        x = torch.tensor([1.0, -5.0, 3.0])
        stats = compute_tensor_stats(x)
        assert math.isclose(stats["max_abs"], 5.0, rel_tol=1e-5)

    def test_empty_tensor_returns_zeros(self):
        x = torch.empty(0)
        stats = compute_tensor_stats(x)
        assert stats["norm"] == 0.0
        assert stats["fraction_nan"] == 0.0

    def test_mixed_nan_fraction(self):
        x = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
        stats = compute_tensor_stats(x)
        assert math.isclose(stats["fraction_nan"], 0.5, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# LayerStatsHook tests
# ---------------------------------------------------------------------------

class TestLayerStatsHook:
    def test_register_and_forward_captures_stats(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        hook = LayerStatsHook("layer0", cfg)
        hook.register(model[0])  # first Linear

        x = torch.randn(2, 8)
        _ = model(x)

        stats = hook.get_stats()
        assert stats["activation"] is not None
        hook.remove()

    def test_get_stats_has_activation_key(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        hook = LayerStatsHook("relu", cfg)
        hook.register(model[1])

        _ = model(torch.randn(2, 8))
        stats = hook.get_stats()
        assert "activation" in stats
        hook.remove()

    def test_get_stats_has_gradient_key(self):
        model = _tiny_model()
        cfg = TrackingConfig()
        hook = LayerStatsHook("layer0", cfg)
        hook.register(model[0])

        _forward_backward(model)
        stats = hook.get_stats()
        assert "gradient" in stats
        hook.remove()

    def test_remove_does_not_raise(self):
        model = _tiny_model()
        cfg = TrackingConfig()
        hook = LayerStatsHook("layer0", cfg)
        hook.register(model[0])
        hook.remove()
        # Double-remove should also be safe
        hook.remove()

    def test_activation_stats_keys_complete(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        hook = LayerStatsHook("layer0", cfg)
        hook.register(model[0])

        _ = model(torch.randn(3, 8))
        act = hook.get_stats()["activation"]
        assert REQUIRED_KEYS <= act.keys()
        hook.remove()


# ---------------------------------------------------------------------------
# ActivationTracker tests
# ---------------------------------------------------------------------------

class TestActivationTracker:
    def test_attach_detach_no_error(self):
        model = _tiny_model()
        cfg = TrackingConfig()
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["0", "1", "2"])
        tracker.detach()

    def test_get_all_stats_has_entries_for_attached_layers(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["0", "2"])

        _ = model(torch.randn(2, 8))
        all_stats = tracker.get_all_stats()
        assert "0" in all_stats
        assert "2" in all_stats
        tracker.detach()

    def test_get_all_stats_returns_dict(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["1"])
        _ = model(torch.randn(2, 8))
        assert isinstance(tracker.get_all_stats(), dict)
        tracker.detach()

    def test_detect_anomalies_returns_list(self):
        model = _tiny_model()
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["0", "1", "2"])
        _ = model(torch.randn(2, 8))
        result = tracker.detect_anomalies()
        assert isinstance(result, list)
        tracker.detach()

    def test_detect_anomalies_catches_nan(self):
        """A layer that outputs NaN should appear in detect_anomalies."""

        class NaNLayer(nn.Module):
            def forward(self, x):
                return torch.full_like(x, float("nan"))

        model = nn.Sequential(nn.Linear(8, 8, bias=False), NaNLayer())
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["1"])

        with torch.no_grad():
            _ = model(torch.randn(2, 8))

        anomalies = tracker.detect_anomalies()
        assert "1" in anomalies
        tracker.detach()

    def test_detect_anomalies_catches_exploding_norm(self):
        """A layer with very large outputs should be flagged."""

        class BigLayer(nn.Module):
            def forward(self, x):
                return x * 1e4  # norm >> 100

        model = nn.Sequential(nn.Linear(8, 8, bias=False), BigLayer())
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["1"])

        with torch.no_grad():
            _ = model(torch.randn(2, 8))

        anomalies = tracker.detect_anomalies()
        assert "1" in anomalies
        tracker.detach()

    def test_detect_anomalies_catches_dead_layer(self):
        """A layer that always outputs a constant (std=0) should be flagged."""

        class DeadLayer(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        model = nn.Sequential(nn.Linear(8, 8, bias=False), DeadLayer())
        cfg = TrackingConfig(track_gradients=False)
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["1"])

        with torch.no_grad():
            _ = model(torch.randn(2, 8))

        anomalies = tracker.detect_anomalies()
        assert "1" in anomalies
        tracker.detach()

    def test_unknown_layer_name_ignored(self):
        """Attaching a non-existent layer name should not raise."""
        model = _tiny_model()
        cfg = TrackingConfig()
        tracker = ActivationTracker(model, cfg)
        tracker.attach(["nonexistent_layer_xyz"])
        all_stats = tracker.get_all_stats()
        assert "nonexistent_layer_xyz" not in all_stats
        tracker.detach()


# ---------------------------------------------------------------------------
# compute_gradient_norm_per_layer tests
# ---------------------------------------------------------------------------

class TestComputeGradientNormPerLayer:
    def test_returns_dict_after_backward(self):
        model = _tiny_model()
        _forward_backward(model)
        result = compute_gradient_norm_per_layer(model)
        assert isinstance(result, dict)

    def test_has_entries_for_all_parameters_with_grad(self):
        model = _tiny_model()
        _forward_backward(model)
        result = compute_gradient_norm_per_layer(model)
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert name in result

    def test_grad_norms_non_negative(self):
        model = _tiny_model()
        _forward_backward(model)
        result = compute_gradient_norm_per_layer(model)
        for name, norm in result.items():
            assert norm >= 0.0, f"Negative norm for {name}: {norm}"

    def test_empty_when_no_backward(self):
        model = _tiny_model()
        # No backward pass — no gradients
        result = compute_gradient_norm_per_layer(model)
        assert result == {}
