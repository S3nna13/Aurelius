"""Tests for src/eval/calibration.py — pure PyTorch, tiny configs."""

from __future__ import annotations

import torch
import pytest

from src.eval.calibration import (
    CalibrationConfig,
    compute_confidence,
    compute_accuracy,
    compute_ece,
    compute_mce,
    reliability_diagram_data,
    TemperatureScaling,
    compute_brier_score,
    ModelCalibrator,
)

# Tiny constants used throughout
N = 50
VOCAB = 8
N_BINS = 5


# ---------------------------------------------------------------------------
# 1. CalibrationConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.n_bins == 10
    assert cfg.min_samples_per_bin == 5
    assert cfg.temperature_init == 1.0


# ---------------------------------------------------------------------------
# 2. compute_confidence — output shape is (N,)
# ---------------------------------------------------------------------------


def test_compute_confidence_shape():
    torch.manual_seed(0)
    logits = torch.randn(N, VOCAB)
    conf = compute_confidence(logits)
    assert conf.shape == (N,)


# ---------------------------------------------------------------------------
# 3. compute_confidence — values in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_confidence_range():
    torch.manual_seed(1)
    logits = torch.randn(N, VOCAB)
    conf = compute_confidence(logits)
    assert (conf >= 0.0).all()
    assert (conf <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. compute_accuracy — output shape is (N,)
# ---------------------------------------------------------------------------


def test_compute_accuracy_shape():
    torch.manual_seed(2)
    logits = torch.randn(N, VOCAB)
    labels = torch.randint(0, VOCAB, (N,))
    acc = compute_accuracy(logits, labels)
    assert acc.shape == (N,)


# ---------------------------------------------------------------------------
# 5. compute_accuracy — values are 0 or 1
# ---------------------------------------------------------------------------


def test_compute_accuracy_binary():
    torch.manual_seed(3)
    logits = torch.randn(N, VOCAB)
    labels = torch.randint(0, VOCAB, (N,))
    acc = compute_accuracy(logits, labels)
    unique = acc.unique()
    for v in unique:
        assert v.item() in (0.0, 1.0)


# ---------------------------------------------------------------------------
# 6. compute_ece — scalar float in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_ece_range():
    torch.manual_seed(4)
    conf = torch.rand(N)
    acc = (torch.rand(N) > 0.5).float()
    ece = compute_ece(conf, acc, n_bins=N_BINS)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# 7. compute_mce — scalar float in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_mce_range():
    torch.manual_seed(5)
    conf = torch.rand(N)
    acc = (torch.rand(N) > 0.5).float()
    mce = compute_mce(conf, acc, n_bins=N_BINS)
    assert isinstance(mce, float)
    assert 0.0 <= mce <= 1.0


# ---------------------------------------------------------------------------
# 8. Perfect calibration gives ECE ≈ 0
# ---------------------------------------------------------------------------


def test_compute_ece_perfect_calibration():
    """When every sample is correct with confidence 1.0, ECE should be 0."""
    conf = torch.ones(N)
    acc = torch.ones(N)
    ece = compute_ece(conf, acc, n_bins=N_BINS)
    assert ece == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. reliability_diagram_data — all 4 required keys present
# ---------------------------------------------------------------------------


def test_reliability_diagram_keys():
    torch.manual_seed(6)
    conf = torch.rand(N)
    acc = (torch.rand(N) > 0.5).float()
    data = reliability_diagram_data(conf, acc, n_bins=N_BINS)
    assert set(data.keys()) == {
        "bin_centers",
        "bin_accuracies",
        "bin_confidences",
        "bin_counts",
    }


# ---------------------------------------------------------------------------
# 10. reliability_diagram_data — bin_counts sums to N
# ---------------------------------------------------------------------------


def test_reliability_diagram_bin_counts_sum():
    torch.manual_seed(7)
    conf = torch.rand(N)
    acc = (torch.rand(N) > 0.5).float()
    data = reliability_diagram_data(conf, acc, n_bins=N_BINS)
    assert data["bin_counts"].sum().item() == pytest.approx(N, abs=1e-3)


# ---------------------------------------------------------------------------
# 11. TemperatureScaling — T=1 leaves logits unchanged
# ---------------------------------------------------------------------------


def test_temperature_scaling_identity():
    torch.manual_seed(8)
    logits = torch.randn(N, VOCAB)
    ts = TemperatureScaling(temperature_init=1.0)
    out = ts(logits)
    assert torch.allclose(out, logits, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. TemperatureScaling.fit — returns a positive float
# ---------------------------------------------------------------------------


def test_temperature_scaling_fit_positive():
    torch.manual_seed(9)
    logits = torch.randn(N, VOCAB)
    labels = torch.randint(0, VOCAB, (N,))
    ts = TemperatureScaling(temperature_init=1.0)
    final_t = ts.fit(logits, labels, n_steps=20, lr=0.05)
    assert isinstance(final_t, float)
    assert final_t > 0.0


# ---------------------------------------------------------------------------
# 13. compute_brier_score — in [0, 1] for multiclass probs
# ---------------------------------------------------------------------------


def test_compute_brier_score_range():
    torch.manual_seed(10)
    logits = torch.randn(N, VOCAB)
    probs = torch.softmax(logits, dim=-1)
    labels = torch.randint(0, VOCAB, (N,))
    score = compute_brier_score(probs, labels)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 14. ModelCalibrator.evaluate — returns all 5 required keys
# ---------------------------------------------------------------------------


def test_model_calibrator_evaluate_keys():
    torch.manual_seed(11)
    logits = torch.randn(N, VOCAB)
    labels = torch.randint(0, VOCAB, (N,))
    cfg = CalibrationConfig(n_bins=N_BINS)
    calibrator = ModelCalibrator(cfg)
    result = calibrator.evaluate(logits, labels)
    assert set(result.keys()) == {"ece", "mce", "brier_score", "mean_confidence", "accuracy"}


# ---------------------------------------------------------------------------
# 15. ECE differs between overconfident and well-calibrated model
# ---------------------------------------------------------------------------


def test_ece_overconfident_vs_calibrated():
    """An overconfident model should have higher ECE than a well-calibrated one."""
    torch.manual_seed(12)
    n = 200

    # Well-calibrated: spread of confidences, accuracy roughly matches confidence
    # Create by forcing argmax to match label half the time with modest confidence
    labels = torch.randint(0, VOCAB, (n,))

    # Calibrated model: logits where argmax == label with ~50% accuracy,
    # moderate confidence
    calibrated_logits = torch.randn(n, VOCAB)
    # Make 50% of samples have argmax == label (moderate calibration)
    for i in range(n // 2):
        calibrated_logits[i, labels[i]] = calibrated_logits[i].max() + 0.5

    cal_conf = compute_confidence(calibrated_logits)
    cal_acc = compute_accuracy(calibrated_logits, labels)
    ece_calibrated = compute_ece(cal_conf, cal_acc, n_bins=N_BINS)

    # Overconfident model: always puts very high probability on one class,
    # but that class is often wrong
    wrong_labels = (labels + 1) % VOCAB
    overconfident_logits = torch.full((n, VOCAB), -10.0)
    overconfident_logits[torch.arange(n), wrong_labels] = 10.0

    over_conf = compute_confidence(overconfident_logits)
    over_acc = compute_accuracy(overconfident_logits, labels)
    ece_overconfident = compute_ece(over_conf, over_acc, n_bins=N_BINS)

    assert ece_overconfident > ece_calibrated
