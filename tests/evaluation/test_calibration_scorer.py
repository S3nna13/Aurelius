"""Tests for src/evaluation/calibration_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math
import pytest

from src.evaluation.calibration_scorer import (
    CalibrationBin,
    CalibrationResult,
    CalibrationScorer,
    CALIBRATION_SCORER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return CalibrationScorer(num_bins=10)


# ---------------------------------------------------------------------------
# CalibrationBin frozen dataclass
# ---------------------------------------------------------------------------

class TestCalibrationBinFrozen:
    def test_is_frozen(self):
        b = CalibrationBin(
            bin_lower=0.0, bin_upper=0.1, count=10,
            mean_confidence=0.05, mean_accuracy=0.05
        )
        with pytest.raises((AttributeError, TypeError)):
            b.count = 5  # type: ignore[misc]

    def test_fields_accessible(self):
        b = CalibrationBin(
            bin_lower=0.2, bin_upper=0.3, count=15,
            mean_confidence=0.25, mean_accuracy=0.20
        )
        assert b.bin_lower == 0.2
        assert b.bin_upper == 0.3
        assert b.count == 15
        assert b.mean_confidence == 0.25
        assert b.mean_accuracy == 0.20

    def test_calibration_error_property_zero_when_perfectly_calibrated(self):
        b = CalibrationBin(
            bin_lower=0.4, bin_upper=0.5, count=5,
            mean_confidence=0.45, mean_accuracy=0.45
        )
        assert math.isclose(b.calibration_error, 0.0, abs_tol=1e-12)

    def test_calibration_error_property_positive_when_miscalibrated(self):
        b = CalibrationBin(
            bin_lower=0.8, bin_upper=0.9, count=10,
            mean_confidence=0.85, mean_accuracy=0.60
        )
        assert math.isclose(b.calibration_error, abs(0.85 - 0.60), rel_tol=1e-9)

    def test_calibration_error_symmetric(self):
        b1 = CalibrationBin(
            bin_lower=0.0, bin_upper=0.1, count=5,
            mean_confidence=0.9, mean_accuracy=0.1
        )
        b2 = CalibrationBin(
            bin_lower=0.0, bin_upper=0.1, count=5,
            mean_confidence=0.1, mean_accuracy=0.9
        )
        assert math.isclose(b1.calibration_error, b2.calibration_error, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# CalibrationResult frozen dataclass
# ---------------------------------------------------------------------------

class TestCalibrationResultFrozen:
    def test_is_frozen(self):
        cr = CalibrationResult(ece=0.1, mce=0.2, bins=[], num_samples=10)
        with pytest.raises((AttributeError, TypeError)):
            cr.ece = 0.0  # type: ignore[misc]

    def test_fields_accessible(self):
        cr = CalibrationResult(ece=0.05, mce=0.15, bins=[], num_samples=100)
        assert cr.ece == 0.05
        assert cr.mce == 0.15
        assert cr.bins == []
        assert cr.num_samples == 100


# ---------------------------------------------------------------------------
# score — basic correctness
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_calibration_result(self, scorer):
        result = scorer.score([0.9, 0.1], [True, False])
        assert isinstance(result, CalibrationResult)

    def test_num_samples_is_total_count(self, scorer):
        confidences = [0.1, 0.5, 0.9]
        correct = [True, False, True]
        result = scorer.score(confidences, correct)
        assert result.num_samples == 3

    def test_bins_count_equals_num_bins(self, scorer):
        confidences = [0.1, 0.5, 0.9]
        correct = [True, False, True]
        result = scorer.score(confidences, correct)
        assert len(result.bins) == 10

    def test_bins_count_with_custom_num_bins(self):
        s = CalibrationScorer(num_bins=5)
        result = s.score([0.1, 0.5, 0.9], [True, False, True])
        assert len(result.bins) == 5

    def test_empty_inputs_ece_zero(self, scorer):
        result = scorer.score([], [])
        assert result.ece == 0.0

    def test_empty_inputs_mce_zero(self, scorer):
        result = scorer.score([], [])
        assert result.mce == 0.0

    def test_empty_inputs_num_samples_zero(self, scorer):
        result = scorer.score([], [])
        assert result.num_samples == 0

    def test_mismatched_lengths_raise_value_error(self, scorer):
        with pytest.raises(ValueError):
            scorer.score([0.5, 0.6], [True])

    def test_perfect_calibration_ece_zero(self):
        # Create a situation where every confidence == actual accuracy in that bin.
        # Put 10 samples all with confidence=0.5 and exactly half correct.
        s = CalibrationScorer(num_bins=10)
        confidences = [0.55] * 10  # all land in the [0.5, 0.6) bin
        correct = [True] * 5 + [False] * 5  # accuracy = 0.5, mean_conf = 0.55
        result = s.score(confidences, correct)
        # ECE should be abs(0.55 - 0.5) * 1.0 — not zero, but let's test a truly
        # perfect case: mean_confidence == mean_accuracy
        # Build a case where they must match exactly
        confidences2 = [0.5] * 5 + [0.5] * 5
        correct2 = [True] * 5 + [False] * 5
        result2 = s.score(confidences2, correct2)
        # mean_conf = 0.5, mean_acc = 0.5 → calibration_error = 0 → ECE = 0
        assert math.isclose(result2.ece, 0.0, abs_tol=1e-12)

    def test_overconfident_ece_positive(self):
        # All samples have high confidence but low accuracy
        s = CalibrationScorer(num_bins=10)
        confidences = [0.95] * 100
        correct = [False] * 100  # accuracy = 0, confidence = 0.95 → large ECE
        result = s.score(confidences, correct)
        assert result.ece > 0.0

    def test_mce_is_max_calibration_error(self, scorer):
        # Two non-empty bins with different calibration errors
        confidences = [0.05] * 10 + [0.95] * 10
        # First bin: conf=0.05, acc=0.0 → error=0.05
        # Last bin: conf=0.95, acc=1.0 → error=0.05
        correct = [False] * 10 + [True] * 10
        result = scorer.score(confidences, correct)
        expected_mce = max(b.calibration_error for b in result.bins if b.count > 0)
        assert math.isclose(result.mce, expected_mce, rel_tol=1e-9)

    def test_single_sample_true(self, scorer):
        result = scorer.score([0.8], [True])
        assert result.num_samples == 1
        assert isinstance(result.ece, float)

    def test_single_sample_false(self, scorer):
        result = scorer.score([0.8], [False])
        assert result.num_samples == 1
        assert result.ece > 0.0  # confidence 0.8, accuracy 0.0

    def test_confidence_one_goes_to_last_bin(self, scorer):
        result = scorer.score([1.0], [True])
        assert result.bins[-1].count == 1

    def test_ece_weighted_sum_of_bin_errors(self, scorer):
        # Manually verify ECE calculation
        confidences = [0.15] * 4 + [0.85] * 6
        correct = [True] * 2 + [False] * 2 + [True] * 6
        result = scorer.score(confidences, correct)
        # Verify ECE matches manual computation from result.bins
        total = result.num_samples
        manual_ece = sum(
            (b.count / total) * b.calibration_error
            for b in result.bins
            if b.count > 0
        )
        assert math.isclose(result.ece, manual_ece, rel_tol=1e-9)

    def test_all_bins_have_bin_lower_bin_upper(self, scorer):
        result = scorer.score([0.5], [True])
        for b in result.bins:
            assert b.bin_lower < b.bin_upper

    def test_bins_cover_zero_to_one(self, scorer):
        result = scorer.score([0.5], [True])
        assert math.isclose(result.bins[0].bin_lower, 0.0, abs_tol=1e-9)
        assert math.isclose(result.bins[-1].bin_upper, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------

class TestReliabilityDiagramData:
    def test_returns_list_of_length_num_bins(self, scorer):
        result = scorer.score([0.1, 0.5, 0.9], [True, False, True])
        data = scorer.reliability_diagram_data(result)
        assert len(data) == 10

    def test_each_entry_has_required_keys(self, scorer):
        result = scorer.score([0.5], [True])
        data = scorer.reliability_diagram_data(result)
        required = {"bin_center", "confidence", "accuracy", "count"}
        for entry in data:
            assert required.issubset(entry.keys())

    def test_bin_center_is_midpoint(self, scorer):
        result = scorer.score([0.5], [True])
        data = scorer.reliability_diagram_data(result)
        for entry, b in zip(data, result.bins):
            expected_center = (b.bin_lower + b.bin_upper) / 2.0
            assert math.isclose(entry["bin_center"], expected_center, rel_tol=1e-9)

    def test_confidence_matches_bin_mean_confidence(self, scorer):
        result = scorer.score([0.5], [True])
        data = scorer.reliability_diagram_data(result)
        for entry, b in zip(data, result.bins):
            assert entry["confidence"] == b.mean_confidence

    def test_accuracy_matches_bin_mean_accuracy(self, scorer):
        result = scorer.score([0.5], [True])
        data = scorer.reliability_diagram_data(result)
        for entry, b in zip(data, result.bins):
            assert entry["accuracy"] == b.mean_accuracy

    def test_count_matches_bin_count(self, scorer):
        result = scorer.score([0.5], [True])
        data = scorer.reliability_diagram_data(result)
        for entry, b in zip(data, result.bins):
            assert entry["count"] == b.count

    def test_total_count_matches_num_samples(self, scorer):
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [True, False, True, False, True]
        result = scorer.score(confidences, correct)
        data = scorer.reliability_diagram_data(result)
        assert sum(entry["count"] for entry in data) == result.num_samples


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in CALIBRATION_SCORER_REGISTRY

    def test_registry_default_is_calibration_scorer_class(self):
        assert CALIBRATION_SCORER_REGISTRY["default"] is CalibrationScorer

    def test_registry_default_instantiable(self):
        cls = CALIBRATION_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, CalibrationScorer)
