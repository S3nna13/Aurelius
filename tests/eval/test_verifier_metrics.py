"""Tests for verifier metrics."""

import pytest
import torch

from src.eval.verifier_metrics import (
    verifier_f1,
    verifier_precision,
    verifier_recall,
    verifier_report,
)


def test_verifier_precision_computes_tp_over_predicted_positive():
    precision = verifier_precision(torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]))
    assert precision.item() == pytest.approx(0.5)


def test_verifier_recall_computes_tp_over_gold_positive():
    recall = verifier_recall(torch.tensor([1, 0, 0]), torch.tensor([1, 1, 0]))
    assert recall.item() == pytest.approx(0.5)


def test_verifier_f1_balances_precision_and_recall():
    f1 = verifier_f1(torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]))
    assert f1.item() == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_verifier_report_aggregates_metrics():
    report = verifier_report(torch.tensor([1, 0, 1]), torch.tensor([1, 1, 1]))
    assert report.accuracy.item() == pytest.approx(2.0 / 3.0)


def test_verifier_precision_handles_no_positive_predictions():
    precision = verifier_precision(torch.tensor([0, 0]), torch.tensor([1, 0]))
    assert precision.item() == pytest.approx(0.0)


def test_verifier_report_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        verifier_report(torch.tensor([1]), torch.tensor([1, 0]))


def test_verifier_f1_zero_when_no_true_positives():
    f1 = verifier_f1(torch.tensor([0, 0]), torch.tensor([1, 1]))
    assert f1.item() == pytest.approx(0.0)
