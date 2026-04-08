"""Tests for acceptance calibration helpers."""

import pytest
import torch

from src.inference.acceptance_calibration import (
    acceptance_brier_score,
    acceptance_ece,
    calibrated_acceptance_report,
)


def test_acceptance_brier_score_zero_for_perfect_predictions():
    score = acceptance_brier_score(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]))
    assert score.item() == pytest.approx(0.0)


def test_acceptance_ece_zero_for_perfect_predictions():
    ece = acceptance_ece(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), n_bins=2)
    assert ece.item() == pytest.approx(0.0)


def test_acceptance_ece_positive_for_miscalibrated_predictions():
    ece = acceptance_ece(torch.tensor([0.9, 0.9]), torch.tensor([0, 0]), n_bins=2)
    assert ece.item() > 0.0


def test_calibrated_acceptance_report_contains_expected_keys():
    report = calibrated_acceptance_report(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
    assert set(report) == {"brier", "ece", "mean_predicted", "mean_accepted"}


def test_acceptance_brier_score_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        acceptance_brier_score(torch.tensor([0.1]), torch.tensor([1, 0]))


def test_acceptance_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        acceptance_ece(torch.tensor([0.1]), torch.tensor([1, 0]))


def test_acceptance_ece_rejects_bad_bins():
    with pytest.raises(ValueError):
        acceptance_ece(torch.tensor([0.1]), torch.tensor([1]), n_bins=0)

