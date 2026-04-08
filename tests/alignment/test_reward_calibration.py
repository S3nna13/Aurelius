"""Tests for reward model calibration."""
import torch
import pytest

from src.alignment.reward_calibration import (
    TemperatureScaling,
    PlattScaling,
    IsotonicCalibration,
    CalibrationEvaluator,
    calibrate_reward_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate synthetic scores and binary labels for calibration tests."""
    torch.manual_seed(42)
    n = 200
    # True probabilities drawn from a Beta-like distribution
    scores = torch.randn(n)  # raw logits (miscalibrated: too extreme)
    # Labels: positively correlated with scores but noisy
    true_probs = torch.sigmoid(scores * 0.5)
    labels = (torch.rand(n) < true_probs).float()
    return scores, labels


# ---------------------------------------------------------------------------
# TemperatureScaling tests
# ---------------------------------------------------------------------------

def test_temperature_scaling_forward():
    """Forward preserves shape and divides by temperature."""
    ts = TemperatureScaling()
    scores = torch.tensor([1.0, 2.0, -1.0, 0.5])
    out = ts(scores)
    assert out.shape == scores.shape, "Output shape must match input shape"
    # With temperature=1.0 (init), output == input
    assert torch.allclose(out, scores), "With T=1, output should equal input"

    # Manually set temperature to 2.0 and verify division
    ts.temperature.data.fill_(2.0)
    out2 = ts(scores)
    assert torch.allclose(out2, scores / 2.0, atol=1e-6), "Output should be score / temperature"


def test_temperature_scaling_fit(synthetic_data):
    """After fit, temperature parameter changes from its initial value of 1.0."""
    scores, labels = synthetic_data
    ts = TemperatureScaling()
    assert ts.temperature.item() == pytest.approx(1.0), "Initial temperature should be 1.0"
    ts.fit(scores, labels)
    # Temperature should have moved from 1.0 after optimization
    assert ts.temperature.item() != pytest.approx(1.0, abs=1e-4), (
        "Temperature should change from 1.0 after fitting"
    )


# ---------------------------------------------------------------------------
# PlattScaling tests
# ---------------------------------------------------------------------------

def test_platt_scaling_forward():
    """Output of PlattScaling.forward is in (0, 1) — probabilities."""
    ps = PlattScaling()
    scores = torch.randn(50)
    probs = ps(scores)
    assert probs.shape == scores.shape, "Output shape must match input shape"
    assert (probs > 0).all(), "All probabilities must be > 0"
    assert (probs < 1).all(), "All probabilities must be < 1"


def test_platt_scaling_fit(synthetic_data):
    """After fit, a and b parameters change from their initial values."""
    scores, labels = synthetic_data
    ps = PlattScaling()

    a_init = ps.a.item()
    b_init = ps.b.item()
    assert a_init == pytest.approx(1.0)
    assert b_init == pytest.approx(0.0)

    ps.fit(scores, labels)

    # At least one of a or b should have changed
    changed = (
        ps.a.item() != pytest.approx(1.0, abs=1e-4)
        or ps.b.item() != pytest.approx(0.0, abs=1e-4)
    )
    assert changed, "At least one of a or b should change after fitting"


# ---------------------------------------------------------------------------
# IsotonicCalibration tests
# ---------------------------------------------------------------------------

def test_isotonic_fit_and_predict(synthetic_data):
    """predict returns values in [0, 1]."""
    scores, labels = synthetic_data
    ic = IsotonicCalibration()
    ic.fit(scores, labels)
    preds = ic.predict(scores)
    assert preds.shape == scores.shape, "Prediction shape must match input shape"
    assert (preds >= 0.0).all(), "All predictions must be >= 0"
    assert (preds <= 1.0).all(), "All predictions must be <= 1"


def test_isotonic_monotone(synthetic_data):
    """predict output is non-decreasing for sorted input scores."""
    scores, labels = synthetic_data
    ic = IsotonicCalibration()
    ic.fit(scores, labels)

    sorted_scores = torch.sort(scores).values
    preds = ic.predict(sorted_scores)

    # Check non-decreasing: each element >= previous
    diffs = preds[1:] - preds[:-1]
    assert (diffs >= -1e-5).all(), (
        "Isotonic predictions must be non-decreasing for sorted inputs"
    )


# ---------------------------------------------------------------------------
# CalibrationEvaluator tests
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    """Perfect calibration (probs == labels for binary case) gives ECE near 0."""
    evaluator = CalibrationEvaluator()
    # Use fractional probabilities that match frequency in each bin
    n = 1000
    torch.manual_seed(0)
    # Create perfectly calibrated probabilities: prob_i = label_i (corner case)
    # A better test: use many samples where each bin's mean(labels) ~ mean(probs)
    probs = torch.linspace(0.05, 0.95, n)
    # Labels sampled to match probs exactly in expectation
    labels = (torch.rand(n) < probs).float()
    ece = evaluator.expected_calibration_error(probs, labels, n_bins=10)
    # With 1000 samples ECE should be small (statistical fluctuation expected)
    assert ece < 0.1, f"ECE for near-perfect calibration should be < 0.1, got {ece:.4f}"


def test_ece_range(synthetic_data):
    """ECE is in [0, 1]."""
    scores, labels = synthetic_data
    probs = torch.sigmoid(scores)
    evaluator = CalibrationEvaluator()
    ece = evaluator.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1], got {ece}"


def test_brier_score_range(synthetic_data):
    """Brier score is in [0, 1]."""
    scores, labels = synthetic_data
    probs = torch.sigmoid(scores)
    evaluator = CalibrationEvaluator()
    bs = evaluator.brier_score(probs, labels)
    assert 0.0 <= bs <= 1.0, f"Brier score must be in [0, 1], got {bs}"


def test_reliability_diagram_keys(synthetic_data):
    """reliability_diagram_data returns dict with correct keys."""
    scores, labels = synthetic_data
    probs = torch.sigmoid(scores)
    evaluator = CalibrationEvaluator()
    result = evaluator.reliability_diagram_data(probs, labels, n_bins=10)

    expected_keys = {"bin_centers", "bin_accuracy", "bin_confidence", "bin_sizes"}
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )
    assert len(result["bin_centers"]) == 10
    assert len(result["bin_accuracy"]) == 10
    assert len(result["bin_confidence"]) == 10
    assert len(result["bin_sizes"]) == 10


# ---------------------------------------------------------------------------
# calibrate_reward_model tests
# ---------------------------------------------------------------------------

def test_calibrate_reward_model_returns_tuple(synthetic_data):
    """calibrate_reward_model returns (calibrator, float, float)."""
    scores, labels = synthetic_data
    result = calibrate_reward_model(
        reward_model=None,
        val_prompts=[],
        val_rewards=scores,
        val_labels=labels,
        method="temperature",
    )
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, "Result must have 3 elements"
    calibrator, ece_before, ece_after = result
    assert isinstance(calibrator, TemperatureScaling), "Calibrator must be TemperatureScaling"
    assert isinstance(ece_before, float), "ece_before must be a float"
    assert isinstance(ece_after, float), "ece_after must be a float"


def test_temperature_calibration_reduces_ece(synthetic_data):
    """After temperature calibration, ECE improves or stays the same."""
    scores, labels = synthetic_data
    _, ece_before, ece_after = calibrate_reward_model(
        reward_model=None,
        val_prompts=[],
        val_rewards=scores,
        val_labels=labels,
        method="temperature",
    )
    # ECE after calibration should be <= ECE before (allowing small tolerance)
    assert ece_after <= ece_before + 0.05, (
        f"Calibration should not significantly worsen ECE: "
        f"before={ece_before:.4f}, after={ece_after:.4f}"
    )
    assert 0.0 <= ece_before <= 1.0
    assert 0.0 <= ece_after <= 1.0
