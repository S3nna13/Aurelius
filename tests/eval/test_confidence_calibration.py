"""Tests for confidence calibration metrics."""

import pytest
import torch

from src.eval.confidence_calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
)


def test_ece_zero_for_perfectly_calibrated_predictions():
    confidences = torch.tensor([0.0, 1.0])
    correctness = torch.tensor([0, 1])
    ece = expected_calibration_error(confidences, correctness, n_bins=2)
    assert ece.item() == pytest.approx(0.0)


def test_brier_score_zero_for_perfect_predictions():
    score = brier_score(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]))
    assert score.item() == pytest.approx(0.0)


def test_calibration_report_aggregates_metrics():
    report = calibration_report(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]), n_bins=2)
    assert report.avg_confidence.item() == pytest.approx(0.5)
    assert report.avg_accuracy.item() == pytest.approx(0.5)


def test_ece_positive_for_miscalibrated_predictions():
    ece = expected_calibration_error(torch.tensor([0.9, 0.9]), torch.tensor([0, 0]), n_bins=2)
    assert ece.item() > 0.0


def test_brier_score_penalizes_overconfidence():
    low = brier_score(torch.tensor([0.6]), torch.tensor([1]))
    high = brier_score(torch.tensor([0.1]), torch.tensor([1]))
    assert high.item() > low.item()


def test_ece_rejects_bad_shapes():
    with pytest.raises(ValueError):
        expected_calibration_error(torch.tensor([0.5]), torch.tensor([1, 0]))


def test_ece_rejects_bad_bin_count():
    with pytest.raises(ValueError):
        expected_calibration_error(torch.tensor([0.5]), torch.tensor([1]), n_bins=0)
