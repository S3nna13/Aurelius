"""Tests for evidence calibration helpers."""

import pytest
import torch

from src.eval.evidence_calibration import (
    evidence_brier_score,
    evidence_calibration_report,
    evidence_ece,
)


def test_evidence_brier_score_zero_for_perfect_predictions():
    score = evidence_brier_score(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]))
    assert score.item() == pytest.approx(0.0)


def test_evidence_ece_zero_for_perfect_predictions():
    ece = evidence_ece(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), n_bins=2)
    assert ece.item() == pytest.approx(0.0)


def test_evidence_ece_positive_for_miscalibrated_predictions():
    ece = evidence_ece(torch.tensor([0.9, 0.9]), torch.tensor([0, 0]), n_bins=2)
    assert ece.item() > 0.0


def test_evidence_calibration_report_contains_keys():
    report = evidence_calibration_report(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
    assert set(report) == {"brier", "ece", "mean_score", "mean_label"}


def test_evidence_brier_score_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evidence_brier_score(torch.tensor([0.1]), torch.tensor([1, 0]))


def test_evidence_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evidence_ece(torch.tensor([0.1]), torch.tensor([1, 0]))


def test_evidence_ece_rejects_bad_bins():
    with pytest.raises(ValueError):
        evidence_ece(torch.tensor([0.1]), torch.tensor([1]), n_bins=0)
