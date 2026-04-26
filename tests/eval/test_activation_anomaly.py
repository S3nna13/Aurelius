"""Tests for src/eval/activation_anomaly.py."""

from __future__ import annotations

import pytest
import torch

from src.eval.activation_anomaly import (
    ActivationMonitor,
    AnomalyConfig,
    AnomalyDetector,
    AnomalyReport,
    compute_activation_statistics,
    compute_layer_similarity,
    detect_dead_neurons,
    detect_exploding_activations,
    detect_nan_inf,
    detect_outlier_activations,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny config / model
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture
def small_input():
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, 4))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AnomalyConfig()
    assert cfg.z_score_threshold == 3.0
    assert cfg.dead_neuron_threshold == 0.01


# ---------------------------------------------------------------------------
# 2. test_detect_nan_inf_clean
# ---------------------------------------------------------------------------


def test_detect_nan_inf_clean():
    t = torch.randn(2, 4, 8)
    has_nan, has_inf = detect_nan_inf(t)
    assert has_nan is False
    assert has_inf is False


# ---------------------------------------------------------------------------
# 3. test_detect_nan_inf_nan
# ---------------------------------------------------------------------------


def test_detect_nan_inf_nan():
    t = torch.randn(2, 4, 8)
    t[0, 0, 0] = float("nan")
    has_nan, has_inf = detect_nan_inf(t)
    assert has_nan is True
    assert has_inf is False


# ---------------------------------------------------------------------------
# 4. test_detect_nan_inf_inf
# ---------------------------------------------------------------------------


def test_detect_nan_inf_inf():
    t = torch.randn(2, 4, 8)
    t[0, 0, 0] = float("inf")
    has_nan, has_inf = detect_nan_inf(t)
    assert has_nan is False
    assert has_inf is True


# ---------------------------------------------------------------------------
# 5. test_detect_exploding_threshold
# ---------------------------------------------------------------------------


def test_detect_exploding_threshold():
    # All values are 200 -- well above threshold of 100
    t = torch.full((2, 4, 8), 200.0)
    frac = detect_exploding_activations(t, threshold=100.0)
    assert frac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. test_detect_exploding_normal
# ---------------------------------------------------------------------------


def test_detect_exploding_normal():
    torch.manual_seed(42)
    t = torch.randn(2, 4, 8)  # values mostly in [-3, 3]
    frac = detect_exploding_activations(t, threshold=100.0)
    assert frac == 0.0


# ---------------------------------------------------------------------------
# 7. test_detect_dead_neurons_all_dead
# ---------------------------------------------------------------------------


def test_detect_dead_neurons_all_dead():
    t = torch.zeros(2, 4, 16)
    frac = detect_dead_neurons(t, threshold=0.01)
    assert frac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. test_detect_dead_neurons_alive
# ---------------------------------------------------------------------------


def test_detect_dead_neurons_alive():
    torch.manual_seed(42)
    t = torch.randn(2, 4, 64)  # random normal -- neurons should be alive
    frac = detect_dead_neurons(t, threshold=0.01)
    assert frac == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# 9. test_compute_activation_statistics_keys
# ---------------------------------------------------------------------------


def test_compute_activation_statistics_keys():
    t = torch.randn(2, 4, 16)
    stats = compute_activation_statistics(t)
    required_keys = {"mean", "std", "min", "max", "kurtosis", "skewness"}
    assert required_keys == set(stats.keys())


# ---------------------------------------------------------------------------
# 10. test_compute_activation_statistics_mean
# ---------------------------------------------------------------------------


def test_compute_activation_statistics_mean():
    # Tensor of all 3.0
    t = torch.full((2, 4, 8), 3.0)
    stats = compute_activation_statistics(t)
    assert stats["mean"] == pytest.approx(3.0, abs=1e-5)
    assert stats["std"] == pytest.approx(0.0, abs=1e-5)
    assert stats["min"] == pytest.approx(3.0, abs=1e-5)
    assert stats["max"] == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 11. test_detect_outlier_activations_range
# ---------------------------------------------------------------------------


def test_detect_outlier_activations_range():
    torch.manual_seed(42)
    t = torch.randn(2, 4, 32)
    frac = detect_outlier_activations(t, baseline_mean=0.0, baseline_std=1.0, z_threshold=3.0)
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# 12. test_compute_layer_similarity_identical
# ---------------------------------------------------------------------------


def test_compute_layer_similarity_identical():
    torch.manual_seed(42)
    t = torch.randn(2, 4, 64)
    sim = compute_layer_similarity(t, t)
    assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 13. test_activation_monitor_captures
# ---------------------------------------------------------------------------


def test_activation_monitor_captures(tiny_model, small_input):
    cfg = AnomalyConfig(layers_to_monitor=[0, 1])
    monitor = ActivationMonitor(tiny_model, cfg)

    monitor.start_monitoring()
    monitor.run_forward(small_input)
    activations = monitor.get_activations()
    monitor.stop_monitoring()

    # Should have captured at least one layer
    assert len(activations) > 0
    # Each entry should be a non-empty list of tensors
    for layer_idx, tensors in activations.items():
        assert len(tensors) > 0
        assert isinstance(tensors[0], torch.Tensor)


# ---------------------------------------------------------------------------
# 14. test_anomaly_detector_analyze_returns_list
# ---------------------------------------------------------------------------


def test_anomaly_detector_analyze_returns_list(tiny_model, small_input):
    cfg = AnomalyConfig(layers_to_monitor=[0, 1], n_calibration_samples=2)
    monitor = ActivationMonitor(tiny_model, cfg)
    detector = AnomalyDetector(cfg)

    reports = detector.analyze(monitor, small_input)
    assert isinstance(reports, list)
    for r in reports:
        assert isinstance(r, AnomalyReport)


# ---------------------------------------------------------------------------
# 15. test_anomaly_detector_summarize_keys
# ---------------------------------------------------------------------------


def test_anomaly_detector_summarize_keys(tiny_model, small_input):
    cfg = AnomalyConfig(layers_to_monitor=[0, 1], n_calibration_samples=2)
    monitor = ActivationMonitor(tiny_model, cfg)
    detector = AnomalyDetector(cfg)

    reports = detector.analyze(monitor, small_input)
    summary = detector.summarize(reports)

    required_keys = {"n_anomalies", "n_nan", "n_exploding", "n_dead", "mean_severity"}
    assert required_keys == set(summary.keys())
    assert isinstance(summary["n_anomalies"], int)
    assert isinstance(summary["mean_severity"], float)
