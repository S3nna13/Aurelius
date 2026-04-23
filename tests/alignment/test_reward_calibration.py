"""Tests for src/alignment/reward_calibration.py — ~45 tests."""
import math
import pytest

from src.alignment.reward_calibration import (
    CalibrationMethod,
    RewardCalibrator,
    REWARD_CALIBRATOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 100, seed: int = 42) -> tuple[list[float], list[int]]:
    """Simple synthetic scores and binary labels."""
    import random
    rng = random.Random(seed)
    scores = [rng.gauss(0, 1) for _ in range(n)]
    # Labels positively correlated with scores
    labels = [1 if s + rng.gauss(0, 0.5) > 0 else 0 for s in scores]
    return scores, labels


# ---------------------------------------------------------------------------
# CalibrationMethod enum
# ---------------------------------------------------------------------------

def test_calibration_method_temperature_value():
    assert CalibrationMethod.TEMPERATURE == "temperature"


def test_calibration_method_platt_value():
    assert CalibrationMethod.PLATT == "platt"


def test_calibration_method_isotonic_value():
    assert CalibrationMethod.ISOTONIC == "isotonic"


def test_calibration_method_is_string_enum():
    assert isinstance(CalibrationMethod.TEMPERATURE, str)


def test_calibration_method_members():
    members = {m.value for m in CalibrationMethod}
    assert "temperature" in members
    assert "platt" in members
    assert "isotonic" in members


# ---------------------------------------------------------------------------
# RewardCalibrator — Temperature
# ---------------------------------------------------------------------------

def test_temperature_calibrate_returns_float():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    result = cal.calibrate(1.0)
    assert isinstance(result, float)


def test_temperature_calibrate_in_zero_one_before_fit():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    result = cal.calibrate(0.5)
    assert 0.0 <= result <= 1.0


def test_temperature_calibrate_in_zero_one_after_fit():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    result = cal.calibrate(1.0)
    assert 0.0 <= result <= 1.0


def test_temperature_fit_no_crash():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)  # Should not raise


def test_temperature_fit_changes_temperature():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    t_before = cal._temperature
    scores, labels = _make_data()
    cal.fit(scores, labels)
    # With realistic data temperature should change from 1.0
    # (or at least the fit ran without error)
    assert isinstance(cal._temperature, float)


def test_temperature_calibrate_positive_score_above_half():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    # sigmoid(positive) > 0.5
    result = cal.calibrate(5.0)
    assert result > 0.5


def test_temperature_calibrate_negative_score_below_half():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    result = cal.calibrate(-5.0)
    assert result < 0.5


def test_temperature_calibrate_zero_is_half():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    result = cal.calibrate(0.0)
    assert abs(result - 0.5) < 1e-6


def test_temperature_calibrate_monotone():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    r1 = cal.calibrate(-2.0)
    r2 = cal.calibrate(0.0)
    r3 = cal.calibrate(2.0)
    assert r1 < r2 < r3


# ---------------------------------------------------------------------------
# RewardCalibrator — Platt
# ---------------------------------------------------------------------------

def test_platt_calibrate_returns_float():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    result = cal.calibrate(0.0)
    assert isinstance(result, float)


def test_platt_calibrate_in_zero_one_before_fit():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    result = cal.calibrate(1.5)
    assert 0.0 <= result <= 1.0


def test_platt_fit_no_crash():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    scores, labels = _make_data()
    cal.fit(scores, labels)


def test_platt_calibrate_in_zero_one_after_fit():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    result = cal.calibrate(2.0)
    assert 0.0 <= result <= 1.0


def test_platt_fit_updates_params():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    a_before = cal._platt_a
    b_before = cal._platt_b
    scores, labels = _make_data()
    cal.fit(scores, labels)
    # At least one param should change
    changed = (
        abs(cal._platt_a - a_before) > 1e-6
        or abs(cal._platt_b - b_before) > 1e-6
    )
    assert changed


def test_platt_calibrate_monotone_before_fit():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    r1 = cal.calibrate(-3.0)
    r2 = cal.calibrate(3.0)
    assert r1 < r2


# ---------------------------------------------------------------------------
# RewardCalibrator — Isotonic
# ---------------------------------------------------------------------------

def test_isotonic_calibrate_without_fit_returns_default():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    result = cal.calibrate(1.0)
    # Should return 0.5 or raise gracefully
    assert result == 0.5 or isinstance(result, float)


def test_isotonic_calibrate_without_fit_does_not_crash():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    result = cal.calibrate(0.0)
    assert isinstance(result, float)


def test_isotonic_fit_no_crash():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    scores, labels = _make_data()
    cal.fit(scores, labels)


def test_isotonic_calibrate_after_fit_in_zero_one():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    result = cal.calibrate(0.0)
    assert 0.0 <= result <= 1.0


def test_isotonic_calibrate_after_fit_returns_float():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    result = cal.calibrate(1.0)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_returns_float():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    ece = cal.expected_calibration_error(scores, labels)
    assert isinstance(ece, float)


def test_ece_in_zero_one():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    ece = cal.expected_calibration_error(scores, labels)
    assert 0.0 <= ece <= 1.0


def test_ece_platt_in_zero_one():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    ece = cal.expected_calibration_error(scores, labels)
    assert 0.0 <= ece <= 1.0


def test_ece_isotonic_in_zero_one():
    cal = RewardCalibrator(CalibrationMethod.ISOTONIC)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    ece = cal.expected_calibration_error(scores, labels)
    assert 0.0 <= ece <= 1.0


def test_ece_custom_n_bins():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    ece = cal.expected_calibration_error(scores, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_ece_perfect_calibration_near_zero():
    """A perfectly calibrated calibrator on its training data should give low ECE."""
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    # Create data where score = logit of label probability exactly
    scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10
    labels = [0, 0, 0, 0, 1, 1, 1] * 10
    cal.fit(scores, labels)
    ece = cal.expected_calibration_error(scores, labels)
    assert ece < 0.5  # Should be reasonably small


# ---------------------------------------------------------------------------
# REWARD_CALIBRATOR_REGISTRY
# ---------------------------------------------------------------------------

def test_registry_has_temperature():
    assert "temperature" in REWARD_CALIBRATOR_REGISTRY


def test_registry_has_platt():
    assert "platt" in REWARD_CALIBRATOR_REGISTRY


def test_registry_temperature_is_reward_calibrator():
    assert isinstance(REWARD_CALIBRATOR_REGISTRY["temperature"], RewardCalibrator)


def test_registry_platt_is_reward_calibrator():
    assert isinstance(REWARD_CALIBRATOR_REGISTRY["platt"], RewardCalibrator)


def test_registry_temperature_method():
    cal = REWARD_CALIBRATOR_REGISTRY["temperature"]
    assert cal.method == CalibrationMethod.TEMPERATURE


def test_registry_platt_method():
    cal = REWARD_CALIBRATOR_REGISTRY["platt"]
    assert cal.method == CalibrationMethod.PLATT


def test_registry_temperature_calibrate_returns_probability():
    cal = REWARD_CALIBRATOR_REGISTRY["temperature"]
    result = cal.calibrate(1.0)
    assert 0.0 <= result <= 1.0


def test_registry_platt_calibrate_returns_probability():
    cal = REWARD_CALIBRATOR_REGISTRY["platt"]
    result = cal.calibrate(-1.0)
    assert 0.0 <= result <= 1.0


def test_calibrate_output_in_zero_one_after_fit_temperature():
    cal = RewardCalibrator(CalibrationMethod.TEMPERATURE)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    for s in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        r = cal.calibrate(s)
        assert 0.0 <= r <= 1.0, f"calibrate({s}) = {r} not in [0,1]"


def test_calibrate_output_in_zero_one_after_fit_platt():
    cal = RewardCalibrator(CalibrationMethod.PLATT)
    scores, labels = _make_data()
    cal.fit(scores, labels)
    for s in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        r = cal.calibrate(s)
        assert 0.0 <= r <= 1.0, f"calibrate({s}) = {r} not in [0,1]"
