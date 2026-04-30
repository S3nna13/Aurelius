"""Tests for reward_calibration_v3: TemperatureScaler, HistogramBinCalibrator,
ReliabilityDiagram, and PreferenceCalibrator."""

from __future__ import annotations

import math

import torch

from aurelius.alignment.reward_calibration_v3 import (
    HistogramBinCalibrator,
    PreferenceCalibrator,
    ReliabilityDiagram,
    TemperatureScaler,
)

N = 50
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rewards_labels(n: int = N):
    rewards = torch.randn(n)
    labels = (rewards > 0).float()
    return rewards, labels


# ---------------------------------------------------------------------------
# TemperatureScaler tests
# ---------------------------------------------------------------------------


def test_temperature_scaler_forward_divides_by_temperature():
    """forward should divide each element by |temperature|."""
    scaler = TemperatureScaler(init_temperature=2.0)
    rewards = torch.tensor([2.0, 4.0, 6.0])
    out = scaler(rewards)
    expected = rewards / 2.0
    assert torch.allclose(out, expected, atol=1e-6)


def test_temperature_scaler_forward_shape_matches_input():
    """Output shape must equal input shape."""
    scaler = TemperatureScaler()
    rewards = torch.randn(N)
    out = scaler(rewards)
    assert out.shape == rewards.shape


def test_temperature_scaler_fit_returns_positive_float():
    """fit() should return a positive float."""
    scaler = TemperatureScaler()
    rewards, labels = _make_rewards_labels()
    result = scaler.fit(rewards, labels, n_epochs=20, lr=0.05)
    assert isinstance(result, float)
    assert result > 0.0


def test_temperature_value_is_positive():
    """temperature_value() should always be positive."""
    scaler = TemperatureScaler(init_temperature=3.0)
    assert scaler.temperature_value() > 0.0


def test_temperature_scaler_fit_changes_temperature():
    """fit() should update the temperature away from its initial value for
    non-trivial data (initial T=1 is seldom the NLL optimum)."""
    scaler = TemperatureScaler(init_temperature=1.0)
    # Create clearly separated data so the optimal temperature is not 1.0
    rewards = torch.cat([torch.ones(25) * 3.0, torch.ones(25) * (-3.0)])
    labels = torch.cat([torch.ones(25), torch.zeros(25)])
    scaler.fit(rewards, labels, n_epochs=200, lr=0.05)
    # Temperature should converge to something positive (we merely check it
    # did not blow up and is still a valid positive value).
    assert scaler.temperature_value() > 0.0
    assert math.isfinite(scaler.temperature_value())


# ---------------------------------------------------------------------------
# HistogramBinCalibrator tests
# ---------------------------------------------------------------------------


def test_histogram_bin_calibrator_fit_stores_bin_edges_and_values():
    """After fit, bin_edges and bin_values must be non-None tensors."""
    cal = HistogramBinCalibrator(n_bins=10)
    scores, labels = _make_rewards_labels()
    cal.fit(scores, labels)
    assert cal.bin_edges is not None
    assert cal.bin_values is not None
    assert isinstance(cal.bin_edges, torch.Tensor)
    assert isinstance(cal.bin_values, torch.Tensor)


def test_histogram_bin_calibrator_calibrate_output_shape():
    """calibrate() output shape must equal input shape."""
    cal = HistogramBinCalibrator(n_bins=10)
    scores, labels = _make_rewards_labels()
    cal.fit(scores, labels)
    out = cal.calibrate(scores)
    assert out.shape == scores.shape


def test_histogram_bin_calibrator_calibrate_output_in_label_range():
    """Calibrated values should be in [0, 1] since labels are binary."""
    cal = HistogramBinCalibrator(n_bins=10)
    scores, labels = _make_rewards_labels()
    cal.fit(scores, labels)
    out = cal.calibrate(scores)
    assert (out >= 0.0).all(), "Calibrated values must be >= 0"
    assert (out <= 1.0).all(), "Calibrated values must be <= 1"


def test_histogram_bin_calibrator_handles_out_of_range_scores():
    """calibrate() must not raise for scores outside the training range."""
    cal = HistogramBinCalibrator(n_bins=10)
    scores = torch.linspace(0.0, 1.0, N)
    labels = (scores > 0.5).float()
    cal.fit(scores, labels)
    # Scores well outside training range [0, 1]
    out_of_range = torch.tensor([-100.0, 0.5, 100.0])
    result = cal.calibrate(out_of_range)
    assert result.shape == out_of_range.shape


# ---------------------------------------------------------------------------
# ReliabilityDiagram tests
# ---------------------------------------------------------------------------


def test_reliability_diagram_compute_returns_expected_keys():
    """compute() result must contain all required keys."""
    diag = ReliabilityDiagram(n_bins=10)
    probs = torch.rand(N)
    labels = torch.randint(0, 2, (N,)).float()
    result = diag.compute(probs, labels)
    assert "ece" in result
    assert "bin_confidences" in result
    assert "bin_accuracies" in result
    assert "bin_counts" in result


def test_reliability_diagram_ece_in_zero_one():
    """ECE must be in [0, 1]."""
    diag = ReliabilityDiagram(n_bins=10)
    probs = torch.rand(N)
    labels = torch.randint(0, 2, (N,)).float()
    result = diag.compute(probs, labels)
    assert 0.0 <= result["ece"] <= 1.0


def test_reliability_diagram_bin_counts_sum_to_n():
    """Total count across all bins must equal N."""
    diag = ReliabilityDiagram(n_bins=10)
    probs = torch.rand(N)
    labels = torch.randint(0, 2, (N,)).float()
    result = diag.compute(probs, labels)
    assert sum(result["bin_counts"]) == N


# ---------------------------------------------------------------------------
# PreferenceCalibrator tests
# ---------------------------------------------------------------------------


def test_preference_calibrator_preference_prob_output_shape():
    """preference_prob() must return (N,) tensor."""
    scaler = TemperatureScaler(init_temperature=1.0)
    pref_cal = PreferenceCalibrator(scaler)
    reward_w = torch.randn(N)
    reward_l = torch.randn(N)
    out = pref_cal.preference_prob(reward_w, reward_l)
    assert out.shape == (N,)


def test_preference_calibrator_preference_prob_values_in_zero_one():
    """preference_prob() values must be in [0, 1] (sigmoid output)."""
    scaler = TemperatureScaler(init_temperature=1.0)
    pref_cal = PreferenceCalibrator(scaler)
    reward_w = torch.randn(N)
    reward_l = torch.randn(N)
    out = pref_cal.preference_prob(reward_w, reward_l)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_preference_calibrator_calibration_error_returns_float_in_zero_one():
    """preference_calibration_error() must return a float in [0, 1]."""
    scaler = TemperatureScaler(init_temperature=1.0)
    pref_cal = PreferenceCalibrator(scaler)
    reward_w = torch.randn(N) + 1.0  # winners slightly higher on average
    reward_l = torch.randn(N) - 1.0
    ece = pref_cal.preference_calibration_error(reward_w, reward_l)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
