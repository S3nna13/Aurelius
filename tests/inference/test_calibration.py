import pytest
import torch
from src.inference.calibration import (
    CalibrationResult,
    expected_calibration_error,
    TemperatureCalibrator,
    TopKCalibrator,
    SequenceCalibrator,
    reliability_diagram_data,
)


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    """When confidence equals accuracy bucket-wise, ECE should be near 0."""
    torch.manual_seed(42)
    n = 1000
    # Generate confidences uniformly in (0, 1)
    confidences = torch.linspace(0.01, 0.99, n)
    # Each sample is "correct" with probability = its confidence
    correctness = (torch.rand(n) < confidences).float()
    result = expected_calibration_error(confidences, correctness, n_bins=10)
    # ECE won't be exactly 0 due to randomness, but should be small
    assert result.ece < 0.1


def test_ece_returns_calibration_result():
    """expected_calibration_error should return a CalibrationResult dataclass."""
    confidences = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    result = expected_calibration_error(confidences, correctness)
    assert isinstance(result, CalibrationResult)
    assert hasattr(result, "ece")
    assert hasattr(result, "mce")
    assert hasattr(result, "brier_score")
    assert hasattr(result, "n_samples")
    assert hasattr(result, "bin_confidences")
    assert hasattr(result, "bin_accuracies")
    assert hasattr(result, "bin_counts")


def test_mce_upper_bounds_ece():
    """MCE >= ECE always (MCE is the max bin error, ECE is the weighted average)."""
    torch.manual_seed(7)
    confidences = torch.rand(200)
    correctness = (torch.rand(200) > 0.5).float()
    result = expected_calibration_error(confidences, correctness, n_bins=10)
    assert result.mce >= result.ece - 1e-6


# ---------------------------------------------------------------------------
# TemperatureCalibrator tests
# ---------------------------------------------------------------------------

def test_temperature_calibrator_init():
    """Temperature should be positive after initialization."""
    cal = TemperatureCalibrator(init_temperature=1.5)
    assert cal.temperature > 0


def test_temperature_calibrator_forward_scales_logits():
    """Output should differ from input when temperature != 1."""
    cal = TemperatureCalibrator(init_temperature=2.0)
    logits = torch.randn(4, 16)
    scaled = cal(logits)
    assert not torch.allclose(scaled, logits)
    assert scaled.shape == logits.shape


def test_temperature_calibrator_fit():
    """fit() should return a list of float losses."""
    torch.manual_seed(0)
    cal = TemperatureCalibrator(init_temperature=1.0)
    logits = torch.randn(50, 32)
    labels = torch.randint(0, 32, (50,))
    losses = cal.fit(logits, labels, n_steps=5, lr=0.01)
    assert isinstance(losses, list)
    assert len(losses) == 5
    assert all(isinstance(l, float) for l in losses)


def test_temperature_lowers_after_overconfident_fit():
    """Overconfident logits (large magnitude) should cause T to increase after fit."""
    torch.manual_seed(1)
    vocab_size = 10
    n = 100
    # Create overconfident logits: large values push probabilities near 0/1
    labels = torch.randint(0, vocab_size, (n,))
    # Scale logits very large to make the model overconfident
    logits = torch.randn(n, vocab_size) * 10.0

    cal = TemperatureCalibrator(init_temperature=1.0)
    t_before = cal.temperature
    cal.fit(logits, labels, n_steps=50, lr=0.05)
    t_after = cal.temperature
    # Overconfident model should require T > 1 to soften predictions
    assert t_after > t_before


# ---------------------------------------------------------------------------
# TopKCalibrator tests
# ---------------------------------------------------------------------------

def test_topk_calibrator_fit_returns_threshold():
    """fit() should return a float threshold in [0, 1]."""
    torch.manual_seed(2)
    confidences = torch.rand(200)
    correctness = (torch.rand(200) > 0.3).float()
    cal = TopKCalibrator(target_precision=0.8)
    tau = cal.fit(confidences, correctness)
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0


# ---------------------------------------------------------------------------
# SequenceCalibrator tests
# ---------------------------------------------------------------------------

def test_sequence_calibrator_mean_log():
    """score() should return a float."""
    cal = SequenceCalibrator(method="mean_log")
    log_probs = [-0.5, -0.3, -0.8, -0.2]
    result = cal.score(log_probs)
    assert isinstance(result, float)


def test_sequence_calibrator_compare():
    """compare() should return a valid index in [0, len(candidates) - 1]."""
    cal = SequenceCalibrator(method="mean_log")
    candidates = [
        [-1.0, -0.5, -0.8],
        [-0.2, -0.1, -0.3],
        [-2.0, -1.5, -1.8],
    ]
    idx = cal.compare(candidates)
    assert isinstance(idx, int)
    assert 0 <= idx < len(candidates)
    # The second candidate has the highest (least negative) mean log prob
    assert idx == 1


# ---------------------------------------------------------------------------
# reliability_diagram_data tests
# ---------------------------------------------------------------------------

def test_reliability_diagram_data_keys():
    """reliability_diagram_data should return dict with required keys."""
    torch.manual_seed(3)
    confidences = torch.rand(100)
    correctness = (torch.rand(100) > 0.5).float()
    data = reliability_diagram_data(confidences, correctness, n_bins=10)
    assert isinstance(data, dict)
    required_keys = {"bin_centers", "bin_accuracies", "bin_confidences", "bin_counts", "ece"}
    assert required_keys.issubset(data.keys())
    assert len(data["bin_centers"]) == 10
    assert len(data["bin_accuracies"]) == 10
    assert len(data["bin_confidences"]) == 10
    assert len(data["bin_counts"]) == 10
    assert isinstance(data["ece"], float)
