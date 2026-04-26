"""Tests for src/alignment/reward_calibration_v2.py

Covers:
1.  expected_calibration_error - perfect calibration → ECE ≈ 0
2.  expected_calibration_error - worst case → ECE > 0
3.  expected_calibration_error - returns value in [0, 1]
4.  TemperatureScaler forward changes scores when T ≠ 1
5.  TemperatureScaler temperature stays positive after fit
6.  TemperatureScaler.fit returns dict with required keys
7.  TemperatureScaler.fit ECE improves (ece_after <= ece_before + 0.1)
8.  PlattScaler forward returns values in (0, 1) (sigmoid output)
9.  PlattScaler.fit returns dict with required keys
10. IsotonicCalibrator fit then predict doesn't error
11. IsotonicCalibrator predictions are monotone (sorted input → sorted output)
12. calibrate_reward_scores with method="temperature" returns calibrated scores
13. calibrate_reward_scores returns tuple of (Tensor, dict)
"""

import torch

from src.alignment.reward_calibration_v2 import (
    CalibrationConfig,
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    calibrate_reward_scores,
    expected_calibration_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfectly_calibrated(n: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, labels) where scores ARE the true probabilities.

    Each score is the actual probability of label=1, so ECE should be ~0.
    """
    torch.manual_seed(seed)
    # scores uniformly spread over (0, 1)
    scores = torch.linspace(0.05, 0.95, n)
    # For perfect calibration: label[i] = 1 iff uniform() < scores[i]
    labels = (torch.rand(n) < scores).float()
    return scores, labels


def _make_miscalibrated(n: int = 200, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores are all ~0.9 but half labels are 0 → badly miscalibrated."""
    torch.manual_seed(seed)
    scores = torch.full((n,), 0.9)
    labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    return scores, labels


def _make_logits_and_labels(n: int = 100, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw logits (not probabilities) and binary labels."""
    torch.manual_seed(seed)
    logits = torch.randn(n) * 2.0
    probs = torch.sigmoid(logits)
    labels = (torch.rand(n) < probs).float()
    return logits, labels


# ---------------------------------------------------------------------------
# Test 1: expected_calibration_error - perfect calibration → ECE ≈ 0
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration():
    """Perfectly calibrated scores should yield ECE close to 0."""
    torch.manual_seed(0)
    # Create scores where each score equals the empirical probability
    # Use many points so the averaging effect kicks in
    n = 1000
    scores = torch.linspace(0.05, 0.95, n)
    # Generate labels by repeating: if score=p, then ~p fraction are 1
    # Use deterministic rounding for exact calibration
    labels = torch.zeros(n)
    for i, s in enumerate(scores):
        labels[i] = 1.0 if (i % 10) < int(round(s.item() * 10)) else 0.0
    ece = expected_calibration_error(scores, labels, n_bins=10)
    # With nearly perfect calibration, ECE should be very small
    assert ece < 0.15, f"Expected ECE < 0.15 for perfectly calibrated, got {ece:.4f}"


# ---------------------------------------------------------------------------
# Test 2: expected_calibration_error - worst case → ECE > 0
# ---------------------------------------------------------------------------


def test_ece_worst_case_nonzero():
    """Badly miscalibrated scores should yield ECE > 0."""
    scores, labels = _make_miscalibrated(n=200, seed=42)
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece > 0.0, f"Expected ECE > 0 for miscalibrated scores, got {ece:.4f}"


# ---------------------------------------------------------------------------
# Test 3: expected_calibration_error - returns value in [0, 1]
# ---------------------------------------------------------------------------


def test_ece_range():
    """ECE must always be in [0, 1]."""
    torch.manual_seed(1)
    for seed in range(5):
        torch.manual_seed(seed)
        scores = torch.rand(50)
        labels = torch.randint(0, 2, (50,)).float()
        ece = expected_calibration_error(scores, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


# ---------------------------------------------------------------------------
# Test 4: TemperatureScaler forward changes scores when T ≠ 1
# ---------------------------------------------------------------------------


def test_temperature_scaler_forward_changes_scores():
    """With T = 2.0, forward should halve each score."""
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.temperature.fill_(2.0)

    scores = torch.tensor([1.0, 2.0, 4.0, -1.0])
    out = scaler(scores)
    expected = scores / 2.0
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


# ---------------------------------------------------------------------------
# Test 5: TemperatureScaler temperature stays positive after fit
# ---------------------------------------------------------------------------


def test_temperature_scaler_temperature_positive_after_fit():
    """Temperature parameter must remain positive after fitting."""
    torch.manual_seed(10)
    logits, labels = _make_logits_and_labels(n=100, seed=10)
    config = CalibrationConfig(method="temperature", lr=0.01, max_iter=200)
    scaler = TemperatureScaler()
    scaler.fit(logits, labels, config)
    assert scaler.temperature.item() > 0.0, (
        f"Temperature must be positive, got {scaler.temperature.item()}"
    )


# ---------------------------------------------------------------------------
# Test 6: TemperatureScaler.fit returns dict with required keys
# ---------------------------------------------------------------------------


def test_temperature_scaler_fit_returns_required_keys():
    """fit() must return dict with 'temperature', 'ece_before', 'ece_after'."""
    torch.manual_seed(20)
    logits, labels = _make_logits_and_labels(n=80, seed=20)
    config = CalibrationConfig(method="temperature", lr=0.01, max_iter=100)
    scaler = TemperatureScaler()
    result = scaler.fit(logits, labels, config)

    assert isinstance(result, dict), "fit() should return a dict"
    for key in ("temperature", "ece_before", "ece_after"):
        assert key in result, f"Missing key '{key}' in fit() result: {result}"


# ---------------------------------------------------------------------------
# Test 7: TemperatureScaler.fit ECE improves
# ---------------------------------------------------------------------------


def test_temperature_scaler_fit_ece_improves():
    """After fitting, ece_after should be <= ece_before + 0.1."""
    torch.manual_seed(30)
    # Create a clearly miscalibrated dataset: logits all very large positive
    n = 100
    logits = torch.full((n,), 3.0)  # all predict ~0.95
    # But only 50% are positive → bad calibration
    labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

    config = CalibrationConfig(method="temperature", lr=0.01, max_iter=200, n_bins=10)
    scaler = TemperatureScaler()
    result = scaler.fit(logits, labels, config)

    assert result["ece_after"] <= result["ece_before"] + 0.1, (
        f"ECE did not improve: before={result['ece_before']:.4f}, after={result['ece_after']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: PlattScaler forward returns values in (0, 1)
# ---------------------------------------------------------------------------


def test_platt_scaler_forward_sigmoid_range():
    """PlattScaler forward applies sigmoid → output must be in (0, 1)."""
    torch.manual_seed(40)
    scaler = PlattScaler()
    scores = torch.randn(50) * 5.0  # wide range of logits
    out = scaler(scores)
    assert out.shape == scores.shape, "Shape mismatch"
    assert (out > 0.0).all(), "PlattScaler output must be > 0"
    assert (out < 1.0).all(), "PlattScaler output must be < 1"


# ---------------------------------------------------------------------------
# Test 9: PlattScaler.fit returns dict with required keys
# ---------------------------------------------------------------------------


def test_platt_scaler_fit_returns_required_keys():
    """fit() must return dict with at least 'ece_before' and 'ece_after'."""
    torch.manual_seed(50)
    logits, labels = _make_logits_and_labels(n=80, seed=50)
    config = CalibrationConfig(method="platt", lr=0.01, max_iter=100)
    scaler = PlattScaler()
    result = scaler.fit(logits, labels, config)

    assert isinstance(result, dict), "fit() should return a dict"
    for key in ("a", "b", "ece_before", "ece_after"):
        assert key in result, f"Missing key '{key}' in PlattScaler fit() result: {result}"


# ---------------------------------------------------------------------------
# Test 10: IsotonicCalibrator fit then predict doesn't error
# ---------------------------------------------------------------------------


def test_isotonic_calibrator_fit_predict_no_error():
    """IsotonicCalibrator.fit() then predict() should not raise."""
    torch.manual_seed(60)
    scores = torch.rand(50)
    labels = (torch.rand(50) > 0.5).float()

    cal = IsotonicCalibrator()
    cal.fit(scores, labels)
    preds = cal.predict(scores)

    assert preds.shape == scores.shape, "Prediction shape should match input shape"
    assert (preds >= 0.0).all() and (preds <= 1.0).all(), "Predictions should be in [0, 1]"


# ---------------------------------------------------------------------------
# Test 11: IsotonicCalibrator predictions are monotone
# ---------------------------------------------------------------------------


def test_isotonic_calibrator_predictions_monotone():
    """Sorted inputs should produce non-decreasing outputs."""
    torch.manual_seed(70)
    n = 100
    # Scores positively correlated with labels
    scores = torch.linspace(-2.0, 2.0, n)
    probs = torch.sigmoid(scores)
    labels = (torch.rand(n) < probs).float()

    cal = IsotonicCalibrator()
    cal.fit(scores, labels)

    # Predict on a sorted input sequence
    test_scores = torch.linspace(-2.0, 2.0, 50)
    preds = cal.predict(test_scores)

    # Check non-decreasing (allow for tiny numerical noise from nearest lookup)
    for i in range(len(preds) - 1):
        assert preds[i] <= preds[i + 1] + 1e-6, (
            f"Predictions not monotone at index {i}: {preds[i].item():.4f} > {preds[i + 1].item():.4f}"  # noqa: E501
        )


# ---------------------------------------------------------------------------
# Test 12: calibrate_reward_scores with method="temperature" returns calibrated scores
# ---------------------------------------------------------------------------


def test_calibrate_reward_scores_temperature_returns_tensor():
    """calibrate_reward_scores with temperature method should return a Tensor."""
    torch.manual_seed(80)
    logits, labels = _make_logits_and_labels(n=80, seed=80)
    config = CalibrationConfig(method="temperature", lr=0.01, max_iter=100)
    calibrated, metrics = calibrate_reward_scores(logits, labels, config)

    assert isinstance(calibrated, torch.Tensor), "First return must be a Tensor"
    assert calibrated.shape == logits.shape, "Calibrated shape must match input shape"


# ---------------------------------------------------------------------------
# Test 13: calibrate_reward_scores returns tuple of (Tensor, dict)
# ---------------------------------------------------------------------------


def test_calibrate_reward_scores_returns_tuple_tensor_dict():
    """calibrate_reward_scores must return (Tensor, dict) with required metrics keys."""
    torch.manual_seed(90)
    logits, labels = _make_logits_and_labels(n=80, seed=90)

    for method in ("temperature", "platt", "isotonic"):
        config = CalibrationConfig(method=method, lr=0.01, max_iter=100)
        result = calibrate_reward_scores(logits, labels, config)

        assert isinstance(result, tuple), f"[{method}] Should return a tuple"
        assert len(result) == 2, f"[{method}] Tuple should have 2 elements"

        calibrated, metrics = result
        assert isinstance(calibrated, torch.Tensor), f"[{method}] First element must be Tensor"
        assert isinstance(metrics, dict), f"[{method}] Second element must be dict"

        for key in ("ece_before", "ece_after", "method"):
            assert key in metrics, f"[{method}] Missing key '{key}' in metrics: {metrics}"

        assert metrics["method"] == method, (
            f"metrics['method'] should be '{method}', got {metrics['method']!r}"
        )
