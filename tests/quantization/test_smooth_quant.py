"""Tests for src/quantization/smooth_quant.py — ≥28 tests, stdlib-only."""

from __future__ import annotations

import pytest

from src.quantization.smooth_quant import (
    SMOOTH_QUANT_REGISTRY,
    SmoothQuantCalibrator,
    SmoothQuantConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(a: float, b: float, rel: float = 1e-6) -> bool:
    """Return True when *a* and *b* are within *rel* relative tolerance."""
    if b == 0.0:
        return abs(a) < 1e-10
    return abs(a - b) / abs(b) < rel


# ---------------------------------------------------------------------------
# SmoothQuantConfig
# ---------------------------------------------------------------------------


class TestSmoothQuantConfig:
    def test_defaults_alpha(self):
        cfg = SmoothQuantConfig()
        assert cfg.alpha == 0.5

    def test_defaults_eps(self):
        cfg = SmoothQuantConfig()
        assert cfg.eps == 1e-5

    def test_custom_alpha(self):
        cfg = SmoothQuantConfig(alpha=0.8)
        assert cfg.alpha == 0.8

    def test_custom_eps(self):
        cfg = SmoothQuantConfig(eps=1e-8)
        assert cfg.eps == 1e-8

    def test_frozen_alpha(self):
        cfg = SmoothQuantConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.alpha = 0.9  # type: ignore[misc]

    def test_frozen_eps(self):
        cfg = SmoothQuantConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.eps = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SmoothQuantCalibrator — calibrate
# ---------------------------------------------------------------------------


class TestCalibrateBasic:
    def setup_method(self):
        self.cal = SmoothQuantCalibrator()

    def test_calibrate_single_channel_alpha_half(self):
        # s = act_max^0.5 / (weight_max^0.5 + eps)
        act = {"c0": 4.0}
        wt = {"c0": 9.0}
        scales = self.cal.calibrate(act, wt)
        expected = (4.0**0.5) / (9.0**0.5 + 1e-5)
        assert approx(scales["c0"], expected)

    def test_calibrate_returns_dict(self):
        scales = self.cal.calibrate({"c0": 2.0}, {"c0": 2.0})
        assert isinstance(scales, dict)

    def test_calibrate_channel_key_preserved(self):
        scales = self.cal.calibrate({"layer.0": 2.0}, {"layer.0": 4.0})
        assert "layer.0" in scales

    def test_calibrate_two_channels(self):
        act = {"c0": 4.0, "c1": 9.0}
        wt = {"c0": 1.0, "c1": 4.0}
        scales = self.cal.calibrate(act, wt)
        assert set(scales.keys()) == {"c0", "c1"}

    def test_calibrate_empty_dict_returns_empty(self):
        scales = self.cal.calibrate({}, {})
        assert scales == {}

    def test_calibrate_missing_weight_channel_skipped(self):
        scales = self.cal.calibrate({"c0": 1.0, "c1": 2.0}, {"c0": 1.0})
        assert "c1" not in scales
        assert "c0" in scales

    def test_calibrate_missing_act_channel_skipped(self):
        scales = self.cal.calibrate({"c0": 1.0}, {"c0": 1.0, "c1": 2.0})
        assert "c1" not in scales

    def test_calibrate_alpha_zero(self):
        # alpha=0 → s = act^0 / (weight^1 + eps) = 1 / (weight_max + eps)
        cal = SmoothQuantCalibrator(SmoothQuantConfig(alpha=0.0, eps=1e-5))
        act = {"c0": 8.0}
        wt = {"c0": 4.0}
        scales = cal.calibrate(act, wt)
        expected = 1.0 / (4.0 + 1e-5)
        assert approx(scales["c0"], expected)

    def test_calibrate_alpha_one(self):
        # alpha=1 → s = act_max^1 / (weight^0 + eps) = act_max / (1 + eps)
        cal = SmoothQuantCalibrator(SmoothQuantConfig(alpha=1.0, eps=1e-5))
        act = {"c0": 6.0}
        wt = {"c0": 3.0}
        scales = cal.calibrate(act, wt)
        expected = 6.0 / (1.0 + 1e-5)
        assert approx(scales["c0"], expected)

    def test_calibrate_eps_prevents_zero_division(self):
        cal = SmoothQuantCalibrator(SmoothQuantConfig(alpha=0.0, eps=1e-5))
        # weight_max = 0 → denominator = 0^1 + eps = eps
        scales = cal.calibrate({"c0": 1.0}, {"c0": 0.0})
        expected = 1.0 / (0.0 + 1e-5)
        assert approx(scales["c0"], expected)

    def test_calibrate_negative_act_uses_abs(self):
        cal = SmoothQuantCalibrator(SmoothQuantConfig(alpha=0.5))
        scales_pos = cal.calibrate({"c": 4.0}, {"c": 4.0})
        scales_neg = cal.calibrate({"c": -4.0}, {"c": 4.0})
        assert approx(scales_pos["c"], scales_neg["c"])


# ---------------------------------------------------------------------------
# SmoothQuantCalibrator — apply_scale
# ---------------------------------------------------------------------------


class TestApplyScale:
    def setup_method(self):
        self.cal = SmoothQuantCalibrator()

    def test_apply_scale_elementwise(self):
        w = [1.0, 2.0, 3.0]
        s = [2.0, 3.0, 4.0]
        result = self.cal.apply_scale(w, s)
        assert result == [2.0, 6.0, 12.0]

    def test_apply_scale_empty(self):
        assert self.cal.apply_scale([], []) == []

    def test_apply_scale_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.cal.apply_scale([1.0, 2.0], [1.0])

    def test_apply_scale_identity(self):
        w = [3.5, -1.2, 0.0]
        result = self.cal.apply_scale(w, [1.0, 1.0, 1.0])
        for r, wv in zip(result, w):
            assert approx(r, wv) or (r == 0.0 and wv == 0.0)

    def test_apply_scale_zero_scale(self):
        result = self.cal.apply_scale([5.0, 3.0], [0.0, 2.0])
        assert result[0] == 0.0
        assert result[1] == 6.0


# ---------------------------------------------------------------------------
# SmoothQuantCalibrator — inverse_scale
# ---------------------------------------------------------------------------


class TestInverseScale:
    def setup_method(self):
        self.cal = SmoothQuantCalibrator()

    def test_inverse_scale_elementwise(self):
        eps = 1e-5
        a = [6.0, 9.0]
        s = [2.0, 3.0]
        result = self.cal.inverse_scale(a, s)
        assert approx(result[0], 6.0 / (2.0 + eps))
        assert approx(result[1], 9.0 / (3.0 + eps))

    def test_inverse_scale_empty(self):
        assert self.cal.inverse_scale([], []) == []

    def test_inverse_scale_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.cal.inverse_scale([1.0], [1.0, 2.0])

    def test_inverse_scale_zero_scale_uses_eps(self):
        eps = 1e-5
        result = self.cal.inverse_scale([1.0], [0.0])
        assert approx(result[0], 1.0 / eps)

    def test_round_trip_apply_then_inverse(self):
        """apply_scale then inverse_scale should approximately recover original."""
        weights = [2.0, 4.0, 8.0]
        scales = [0.5, 2.0, 4.0]
        smoothed_w = self.cal.apply_scale(weights, scales)
        # inverse_scale divides by s + eps; apply_scale multiplied by s exactly
        # We verify that inverse_scale(apply_scale(w, s), s) ≈ w (up to eps)
        recovered = self.cal.inverse_scale(smoothed_w, scales)
        for orig, rec in zip(weights, recovered):
            assert abs(orig - rec) < orig * 1e-4 + 1e-6


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(SMOOTH_QUANT_REGISTRY, dict)

    def test_default_key_present(self):
        assert "default" in SMOOTH_QUANT_REGISTRY

    def test_default_maps_to_calibrator_class(self):
        assert SMOOTH_QUANT_REGISTRY["default"] is SmoothQuantCalibrator

    def test_registry_instantiable(self):
        cls = SMOOTH_QUANT_REGISTRY["default"]
        obj = cls()
        assert isinstance(obj, SmoothQuantCalibrator)
