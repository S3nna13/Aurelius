"""Tests for F1 score evaluator."""

from __future__ import annotations

from src.evaluation.f1_score import F1Score


class TestF1Score:
    def test_perfect_score(self):
        f1 = F1Score()
        result = f1.compute(true_pos=10, false_pos=0, false_neg=0)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_half_precision(self):
        f1 = F1Score()
        result = f1.compute(true_pos=5, false_pos=5, false_neg=0)
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0

    def test_zero_division(self):
        f1 = F1Score()
        result = f1.compute(true_pos=0, false_pos=0, false_neg=0)
        assert result["f1"] == 0.0

    def test_batch_perfect(self):
        f1 = F1Score()
        result = f1.compute_batch([1, 1, 0, 0], [1, 1, 0, 0])
        assert result["f1"] == 1.0

    def test_batch_mixed(self):
        f1 = F1Score()
        result = f1.compute_batch([1, 1, 0, 0], [1, 0, 0, 0])
        assert result["f1"] > 0.0
