"""Tests for src/eval/calibration_suite.py — 16 tests."""

from __future__ import annotations

import pytest
import torch

from aurelius.eval.calibration_suite import (
    CalibrationBins,
    ReliabilityDiagram,
    BrierScore,
    CalibrationEvaluator,
)

N = 100
N_BINS = 10


# ---------------------------------------------------------------------------
# CalibrationBins
# ---------------------------------------------------------------------------


def test_bin_returns_correct_keys():
    """Test 1: CalibrationBins.bin returns dict with correct keys."""
    cb = CalibrationBins(n_bins=N_BINS)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    result = cb.bin(confidences, correct)
    assert "bin_confidences" in result
    assert "bin_accuracies" in result
    assert "bin_counts" in result


def test_bin_counts_sum_to_n():
    """Test 2: Bin counts sum to N."""
    cb = CalibrationBins(n_bins=N_BINS)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    result = cb.bin(confidences, correct)
    assert result["bin_counts"].sum().item() == N


def test_perfect_calibration_ece_zero():
    """Test 3: Perfect calibration — conf == acc per bin → ECE ≈ 0."""
    n_bins = 5
    cb = CalibrationBins(n_bins=n_bins)
    # Put 20 samples exactly at the centre of each bin [0, 0.2, 0.4, 0.6, 0.8]
    # bin 0: [0, 0.2) → conf = 0.1, acc = 0.1  (2/20 correct)
    # bin 1: [0.2, 0.4) → conf = 0.3, acc = 0.3 (6/20 correct)
    # ...etc.
    # Simpler: build a dataset where mean conf == mean acc per bin.
    # Use n_bins=1 so one big bin; set acc = mean confidence.
    cb1 = CalibrationBins(n_bins=1)
    confidences = torch.full((10,), 0.7)
    # 7 out of 10 correct → acc = 0.7 = conf
    correct = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    ece = cb1.ece(confidences, correct)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_all_wrong_ece_large():
    """Test 4: All wrong predictions → ECE is large."""
    cb = CalibrationBins(n_bins=1)
    # conf = 1.0, acc = 0.0 → ECE = 1.0
    confidences = torch.ones(N)
    correct = torch.zeros(N)
    ece = cb.ece(confidences, correct)
    assert ece > 0.5


def test_ece_in_unit_interval():
    """Test 5: ECE in [0, 1]."""
    cb = CalibrationBins(n_bins=N_BINS)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    ece = cb.ece(confidences, correct)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# ReliabilityDiagram
# ---------------------------------------------------------------------------


def test_reliability_diagram_returns_required_keys():
    """Test 6: ReliabilityDiagram.compute returns 'ece', 'mce', 'overconfident_fraction'."""
    cb = CalibrationBins(n_bins=N_BINS)
    rd = ReliabilityDiagram(bins=cb)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    result = rd.compute(confidences, correct)
    assert "ece" in result
    assert "mce" in result
    assert "overconfident_fraction" in result


def test_overconfident_fraction_in_unit_interval():
    """Test 7: Overconfident fraction in [0, 1]."""
    cb = CalibrationBins(n_bins=N_BINS)
    rd = ReliabilityDiagram(bins=cb)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    result = rd.compute(confidences, correct)
    assert 0.0 <= result["overconfident_fraction"] <= 1.0


def test_mce_geq_ece():
    """Test 8: MCE >= ECE always."""
    cb = CalibrationBins(n_bins=N_BINS)
    rd = ReliabilityDiagram(bins=cb)
    torch.manual_seed(42)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,))
    result = rd.compute(confidences, correct)
    assert result["mce"] >= result["ece"] - 1e-7


# ---------------------------------------------------------------------------
# BrierScore
# ---------------------------------------------------------------------------


def test_brier_score_binary_perfect():
    """Test 9: BrierScore binary — perfect predictions → score = 0."""
    bs = BrierScore()
    probs = torch.tensor([1.0, 0.0, 1.0, 0.0])
    labels = torch.tensor([1, 0, 1, 0])
    score = bs(probs, labels)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_brier_score_binary_all_wrong():
    """Test 10: BrierScore binary — all wrong (p=1, y=0) → score = 1."""
    bs = BrierScore()
    probs = torch.ones(N)
    labels = torch.zeros(N, dtype=torch.long)
    score = bs(probs, labels)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_brier_score_multiclass_range():
    """Test 11: BrierScore multiclass output in [0, 2]."""
    bs = BrierScore()
    torch.manual_seed(0)
    n, c = 50, 5
    probs = torch.softmax(torch.randn(n, c), dim=-1)
    labels = torch.randint(0, c, (n,))
    score = bs(probs, labels)
    assert 0.0 <= score <= 2.0


def test_brier_score_shape_binary():
    """Test 12: BrierScore shape check — (N,) input for binary."""
    bs = BrierScore()
    probs = torch.rand(30)
    labels = torch.randint(0, 2, (30,))
    score = bs(probs, labels)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------


def test_evaluator_returns_all_required_keys():
    """Test 13: CalibrationEvaluator.evaluate returns all required keys."""
    ev = CalibrationEvaluator(n_bins=N_BINS)
    logits = torch.randn(N, 5)
    labels = torch.randint(0, 5, (N,))
    result = ev.evaluate(logits, labels)
    for key in ("ece", "mce", "brier", "overconfident_fraction", "n_samples"):
        assert key in result


def test_evaluator_perfect_predictor():
    """Test 14: evaluate with perfect predictor — ECE ≈ 0, Brier ≈ 0."""
    ev = CalibrationEvaluator(n_bins=10)
    n, c = 200, 4
    # Perfect logits: huge value on correct class
    labels = torch.randint(0, c, (n,))
    logits = torch.full((n, c), -1e6)
    logits[torch.arange(n), labels] = 1e6
    result = ev.evaluate(logits, labels)
    assert result["ece"] == pytest.approx(0.0, abs=1e-4)
    assert result["brier"] == pytest.approx(0.0, abs=1e-4)


def test_evaluator_n_samples_correct():
    """Test 15: evaluate n_samples correct."""
    ev = CalibrationEvaluator(n_bins=N_BINS)
    n = 77
    logits = torch.randn(n, 3)
    labels = torch.randint(0, 3, (n,))
    result = ev.evaluate(logits, labels)
    assert result["n_samples"] == float(n)


def test_ece_with_one_bin():
    """Test 16: ECE with n_bins=1 equals |mean_conf - mean_acc|."""
    cb = CalibrationBins(n_bins=1)
    torch.manual_seed(7)
    confidences = torch.rand(N)
    correct = torch.randint(0, 2, (N,)).float()
    ece = cb.ece(confidences, correct)
    expected = abs(confidences.mean().item() - correct.mean().item())
    assert ece == pytest.approx(expected, abs=1e-6)
