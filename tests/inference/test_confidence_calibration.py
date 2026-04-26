"""Tests for token-level confidence calibration."""

import pytest
import torch
import torch.nn.functional as F

from src.inference.confidence_calibration import (
    CalibrationConfig,
    CalibrationEvaluator,
    TemperatureScaler,
    compute_ece,
    compute_reliability_diagram,
    compute_token_confidence,
    temperature_scale,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# CalibrationConfig tests
# ---------------------------------------------------------------------------


def test_calibration_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.n_bins == 10
    assert cfg.temperature == 1.0
    assert cfg.use_ece is True
    assert cfg.smoothing == 0.0


def test_calibration_config_custom():
    cfg = CalibrationConfig(n_bins=20, temperature=2.0, use_ece=False, smoothing=0.1)
    assert cfg.n_bins == 20
    assert cfg.temperature == 2.0
    assert cfg.use_ece is False
    assert cfg.smoothing == 0.1


# ---------------------------------------------------------------------------
# temperature_scale tests
# ---------------------------------------------------------------------------


def test_temperature_scale_identity():
    """T=1.0 should return unchanged logits."""
    logits = torch.randn(2, 8, 32)
    scaled = temperature_scale(logits, 1.0)
    assert torch.allclose(scaled, logits)


def test_temperature_scale_shape_preserved():
    """Output shape must match input shape."""
    logits = torch.randn(3, 5, 16)
    scaled = temperature_scale(logits, 2.0)
    assert scaled.shape == logits.shape


def test_temperature_scale_reduces_confidence():
    """T > 1 should produce a softer (lower max-prob) distribution."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 64)
    conf_before = F.softmax(logits, dim=-1).max(dim=-1).values
    conf_after = F.softmax(temperature_scale(logits, 2.0), dim=-1).max(dim=-1).values
    assert (conf_after < conf_before).all()


def test_temperature_scale_halves_logits():
    """T=2.0 should exactly halve the logit values."""
    logits = torch.tensor([2.0, 4.0, 6.0])
    scaled = temperature_scale(logits, 2.0)
    assert torch.allclose(scaled, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# compute_token_confidence tests
# ---------------------------------------------------------------------------


def test_compute_token_confidence_shape():
    """Output shape should be (B, T)."""
    logits = torch.randn(2, 10, 64)
    conf = compute_token_confidence(logits)
    assert conf.shape == (2, 10)


def test_compute_token_confidence_range():
    """All confidence values must lie in [0, 1]."""
    torch.manual_seed(1)
    logits = torch.randn(4, 8, 128)
    conf = compute_token_confidence(logits)
    assert (conf >= 0).all()
    assert (conf <= 1.0 + 1e-6).all()


def test_compute_token_confidence_is_max_prob():
    """confidence should equal max of softmax distribution."""
    logits = torch.randn(1, 3, 16)
    conf = compute_token_confidence(logits)
    expected = F.softmax(logits, dim=-1).max(dim=-1).values
    assert torch.allclose(conf, expected)


def test_compute_token_confidence_deterministic():
    """A one-hot logit vector should give confidence = 1."""
    logits = torch.zeros(1, 1, 8)
    logits[0, 0, 3] = 100.0  # near one-hot
    conf = compute_token_confidence(logits)
    assert conf[0, 0].item() > 0.99


# ---------------------------------------------------------------------------
# compute_ece tests
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated():
    """Perfectly calibrated model should yield ECE near 0."""
    torch.manual_seed(7)
    n = 2000
    confidences = torch.linspace(0.01, 0.99, n)
    correct = (torch.rand(n) < confidences).float()
    ece = compute_ece(confidences, correct, n_bins=10)
    assert ece < 0.1


def test_ece_overconfident():
    """Overconfident model (high conf, low accuracy) should have ECE > 0."""
    n = 500
    # All confidences high but only 50% accurate
    confidences = torch.full((n,), 0.95)
    correct = (torch.rand(n) > 0.5).float()
    ece = compute_ece(confidences, correct, n_bins=10)
    assert ece > 0.0


def test_ece_returns_float():
    confidences = torch.rand(100)
    correct = (torch.rand(100) > 0.5).float()
    ece = compute_ece(confidences, correct, n_bins=10)
    assert isinstance(ece, float)


def test_ece_nonnegative():
    torch.manual_seed(3)
    confidences = torch.rand(200)
    correct = (torch.rand(200) > 0.5).float()
    ece = compute_ece(confidences, correct, n_bins=10)
    assert ece >= 0.0


# ---------------------------------------------------------------------------
# compute_reliability_diagram tests
# ---------------------------------------------------------------------------


def test_reliability_diagram_bin_count():
    """All three lists must have exactly n_bins elements."""
    torch.manual_seed(4)
    confidences = torch.rand(300)
    correct = (torch.rand(300) > 0.5).float()
    result = compute_reliability_diagram(confidences, correct, n_bins=10)
    assert len(result["bin_confidences"]) == 10
    assert len(result["bin_accuracies"]) == 10
    assert len(result["bin_counts"]) == 10


def test_reliability_diagram_keys():
    confidences = torch.rand(50)
    correct = (torch.rand(50) > 0.5).float()
    result = compute_reliability_diagram(confidences, correct, n_bins=5)
    assert set(result.keys()) == {"bin_confidences", "bin_accuracies", "bin_counts"}


def test_reliability_diagram_counts_sum_to_n():
    """Sum of bin_counts must equal the total number of samples."""
    n = 150
    confidences = torch.rand(n)
    correct = (torch.rand(n) > 0.5).float()
    result = compute_reliability_diagram(confidences, correct, n_bins=10)
    assert sum(result["bin_counts"]) == n


# ---------------------------------------------------------------------------
# TemperatureScaler tests
# ---------------------------------------------------------------------------


def test_temperature_scaler_default_temperature():
    """Default temperature should come from config."""
    cfg = CalibrationConfig(temperature=1.0)
    scaler = TemperatureScaler(cfg)
    assert scaler.temperature == 1.0


def test_temperature_scaler_fit_valid_range():
    """fit() must select a temperature from the predefined grid."""
    torch.manual_seed(5)
    cfg = CalibrationConfig()
    scaler = TemperatureScaler(cfg)
    logits_list = [torch.randn(20, 64)]
    labels_list = [torch.randint(0, 64, (20,))]
    scaler.fit(logits_list, labels_list)
    assert scaler.temperature in TemperatureScaler._GRID


def test_temperature_scaler_transform_changes_scale():
    """transform() should differ from the original logits when T != 1."""
    cfg = CalibrationConfig(temperature=2.0)
    scaler = TemperatureScaler(cfg)
    logits = torch.randn(4, 32)
    transformed = scaler.transform(logits)
    assert not torch.allclose(transformed, logits)
    assert transformed.shape == logits.shape


def test_temperature_scaler_fit_multiple_batches():
    """fit() should work with multiple (logits, labels) pairs."""
    torch.manual_seed(6)
    cfg = CalibrationConfig()
    scaler = TemperatureScaler(cfg)
    logits_list = [torch.randn(10, 32) for _ in range(3)]
    labels_list = [torch.randint(0, 32, (10,)) for _ in range(3)]
    scaler.fit(logits_list, labels_list)
    assert scaler.temperature in TemperatureScaler._GRID


# ---------------------------------------------------------------------------
# CalibrationEvaluator tests
# ---------------------------------------------------------------------------


def test_calibration_evaluator_returns_dict(small_model):
    """evaluate() must return a dict with ece, mean_confidence, accuracy."""
    cfg = CalibrationConfig(n_bins=5)
    evaluator = CalibrationEvaluator(cfg)
    data = [torch.randint(0, 256, (2, 8))]
    result = evaluator.evaluate(small_model, iter(data))
    assert isinstance(result, dict)
    assert "ece" in result
    assert "mean_confidence" in result
    assert "accuracy" in result


def test_calibration_evaluator_ece_nonneg(small_model):
    """ECE must be non-negative."""
    evaluator = CalibrationEvaluator()
    data = [torch.randint(0, 256, (1, 6))]
    result = evaluator.evaluate(small_model, iter(data))
    assert result["ece"] >= 0.0


def test_calibration_evaluator_confidence_range(small_model):
    """mean_confidence must be in [0, 1]."""
    evaluator = CalibrationEvaluator()
    data = [torch.randint(0, 256, (2, 10))]
    result = evaluator.evaluate(small_model, iter(data))
    assert 0.0 <= result["mean_confidence"] <= 1.0


def test_calibration_evaluator_accuracy_range(small_model):
    """accuracy must be in [0, 1]."""
    evaluator = CalibrationEvaluator()
    data = [torch.randint(0, 256, (2, 10))]
    result = evaluator.evaluate(small_model, iter(data))
    assert 0.0 <= result["accuracy"] <= 1.0


def test_calibration_evaluator_empty_iter(small_model):
    """evaluate() on an empty iterator should return zero metrics."""
    evaluator = CalibrationEvaluator()
    result = evaluator.evaluate(small_model, iter([]))
    assert result["ece"] == 0.0
    assert result["mean_confidence"] == 0.0
    assert result["accuracy"] == 0.0
