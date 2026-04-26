"""Tests for GPTQ-style calibration module."""

from __future__ import annotations

import pytest

from src.quantization.gptq_calibration import (
    CalibrationStats,
    GPTQCalibrator,
    GPTQConfig,
)

# ---------------------------------------------------------------------------
# GPTQConfig defaults
# ---------------------------------------------------------------------------


class TestGPTQConfigDefaults:
    def test_bits_default(self):
        cfg = GPTQConfig()
        assert cfg.bits == 4

    def test_group_size_default(self):
        cfg = GPTQConfig()
        assert cfg.group_size == 128

    def test_actorder_default(self):
        cfg = GPTQConfig()
        assert cfg.actorder is False

    def test_damp_percent_default(self):
        cfg = GPTQConfig()
        assert cfg.damp_percent == 0.01

    def test_custom_bits(self):
        cfg = GPTQConfig(bits=8)
        assert cfg.bits == 8

    def test_custom_group_size(self):
        cfg = GPTQConfig(group_size=64)
        assert cfg.group_size == 64

    def test_custom_actorder(self):
        cfg = GPTQConfig(actorder=True)
        assert cfg.actorder is True

    def test_custom_damp_percent(self):
        cfg = GPTQConfig(damp_percent=0.05)
        assert cfg.damp_percent == 0.05


# ---------------------------------------------------------------------------
# CalibrationStats fields
# ---------------------------------------------------------------------------


class TestCalibrationStatsFields:
    def test_has_layer_name(self):
        s = CalibrationStats(
            layer_name="fc1", n_samples=10, input_mean=0.0, input_std=1.0, hessian_diag=[1.0]
        )
        assert s.layer_name == "fc1"

    def test_has_n_samples(self):
        s = CalibrationStats(
            layer_name="fc1", n_samples=10, input_mean=0.0, input_std=1.0, hessian_diag=[1.0]
        )
        assert s.n_samples == 10

    def test_has_input_mean(self):
        s = CalibrationStats(
            layer_name="fc1", n_samples=10, input_mean=2.5, input_std=1.0, hessian_diag=[1.0]
        )
        assert s.input_mean == 2.5

    def test_has_input_std(self):
        s = CalibrationStats(
            layer_name="fc1", n_samples=10, input_mean=0.0, input_std=3.0, hessian_diag=[1.0]
        )
        assert s.input_std == 3.0

    def test_has_hessian_diag(self):
        s = CalibrationStats(
            layer_name="fc1", n_samples=10, input_mean=0.0, input_std=1.0, hessian_diag=[1.0, 2.0]
        )
        assert s.hessian_diag == [1.0, 2.0]


# ---------------------------------------------------------------------------
# GPTQCalibrator.accumulate
# ---------------------------------------------------------------------------


class TestAccumulate:
    def test_accumulate_no_crash(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 2.0], [3.0, 4.0]])

    def test_accumulate_empty_inputs_no_crash(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [])

    def test_accumulate_single_sample(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 0.0, -1.0]])
        stats = cal.finalize("layer1")
        assert stats.n_samples == 1

    def test_accumulate_multiple_calls(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 2.0]])
        cal.accumulate("layer1", [[3.0, 4.0], [5.0, 6.0]])
        stats = cal.finalize("layer1")
        assert stats.n_samples == 3

    def test_accumulate_uses_default_config(self):
        cal = GPTQCalibrator()
        assert cal.config.bits == 4

    def test_accumulate_with_custom_config(self):
        cfg = GPTQConfig(bits=8)
        cal = GPTQCalibrator(config=cfg)
        cal.accumulate("layer1", [[1.0]])

    def test_accumulate_stores_per_layer(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer_a", [[1.0]])
        cal.accumulate("layer_b", [[2.0, 3.0]])
        stats_a = cal.finalize("layer_a")
        stats_b = cal.finalize("layer_b")
        assert stats_a.layer_name == "layer_a"
        assert stats_b.layer_name == "layer_b"


# ---------------------------------------------------------------------------
# GPTQCalibrator.finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_returns_calibration_stats(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 2.0]])
        result = cal.finalize("layer1")
        assert isinstance(result, CalibrationStats)

    def test_finalize_hessian_diag_is_list(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 2.0, 3.0]])
        stats = cal.finalize("layer1")
        assert isinstance(stats.hessian_diag, list)

    def test_finalize_hessian_diag_length_matches_features(self):
        cal = GPTQCalibrator()
        d = 5
        cal.accumulate("layer1", [[float(i) for i in range(d)]])
        stats = cal.finalize("layer1")
        assert len(stats.hessian_diag) == d

    def test_finalize_hessian_diag_length_larger(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0] * 10])
        stats = cal.finalize("layer1")
        assert len(stats.hessian_diag) == 10

    def test_finalize_missing_layer_raises(self):
        cal = GPTQCalibrator()
        with pytest.raises(KeyError):
            cal.finalize("nonexistent")

    def test_finalize_n_samples_correct(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0]] * 7)
        stats = cal.finalize("layer1")
        assert stats.n_samples == 7

    def test_finalize_layer_name_preserved(self):
        cal = GPTQCalibrator()
        cal.accumulate("my_layer", [[1.0]])
        stats = cal.finalize("my_layer")
        assert stats.layer_name == "my_layer"

    def test_finalize_hessian_diag_values_are_mean_squares(self):
        cal = GPTQCalibrator()
        # Two samples: [2.0] and [4.0] -> mean(x^2) = (4 + 16)/2 = 10
        cal.accumulate("layer1", [[2.0], [4.0]])
        stats = cal.finalize("layer1")
        assert abs(stats.hessian_diag[0] - 10.0) < 1e-9

    def test_finalize_input_std_nonneg(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[1.0, 2.0], [3.0, 4.0]])
        stats = cal.finalize("layer1")
        assert stats.input_std >= 0.0

    def test_finalize_input_mean_scalar(self):
        cal = GPTQCalibrator()
        cal.accumulate("layer1", [[2.0, 4.0]])
        stats = cal.finalize("layer1")
        assert isinstance(stats.input_mean, float)


# ---------------------------------------------------------------------------
# quantize_weight
# ---------------------------------------------------------------------------


class TestQuantizeWeight:
    def test_values_in_range_4bit(self):
        cal = GPTQCalibrator()
        w = [0.5, -0.5, 1.0, -1.0, 0.0]
        scale, zp = cal.compute_scale(w, bits=4)
        q = cal.quantize_weight(w, scale, zp, bits=4)
        max_val = (1 << 4) - 1
        for v in q:
            assert 0 <= v <= max_val

    def test_values_in_range_8bit(self):
        cal = GPTQCalibrator()
        w = list(range(-10, 11))
        scale, zp = cal.compute_scale(w, bits=8)
        q = cal.quantize_weight([float(x) for x in w], scale, zp, bits=8)
        max_val = (1 << 8) - 1
        for v in q:
            assert 0 <= v <= max_val

    def test_values_in_range_2bit(self):
        cal = GPTQCalibrator()
        w = [0.3, -0.3, 0.9, -0.9]
        scale, zp = cal.compute_scale(w, bits=2)
        q = cal.quantize_weight(w, scale, zp, bits=2)
        max_val = (1 << 2) - 1
        for v in q:
            assert 0 <= v <= max_val

    def test_returns_list_of_int(self):
        cal = GPTQCalibrator()
        w = [1.0, 2.0, 3.0]
        scale, zp = cal.compute_scale(w, bits=4)
        q = cal.quantize_weight(w, scale, zp, bits=4)
        assert isinstance(q, list)
        for v in q:
            assert isinstance(v, int)

    def test_length_preserved(self):
        cal = GPTQCalibrator()
        w = [0.1, 0.2, 0.3, 0.4, 0.5]
        scale, zp = cal.compute_scale(w, bits=4)
        q = cal.quantize_weight(w, scale, zp, bits=4)
        assert len(q) == len(w)

    def test_clamping_prevents_overflow(self):
        cal = GPTQCalibrator()
        # Very large weight with small scale
        q = cal.quantize_weight([1e10], scale=1.0, zero_point=0.0, bits=4)
        assert q[0] == (1 << 4) - 1

    def test_clamping_prevents_underflow(self):
        cal = GPTQCalibrator()
        q = cal.quantize_weight([-1e10], scale=1.0, zero_point=0.0, bits=4)
        assert q[0] == 0

    def test_zero_weight_near_zero_point(self):
        cal = GPTQCalibrator()
        # zero_point=0, scale=1.0: w=0 -> q=0
        q = cal.quantize_weight([0.0], scale=1.0, zero_point=0.0, bits=4)
        assert q[0] == 0


# ---------------------------------------------------------------------------
# dequantize
# ---------------------------------------------------------------------------


class TestDequantize:
    def test_returns_list_of_float(self):
        cal = GPTQCalibrator()
        dq = cal.dequantize([0, 5, 10], scale=1.0, zero_point=0.0)
        assert isinstance(dq, list)
        for v in dq:
            assert isinstance(v, float)

    def test_length_preserved(self):
        cal = GPTQCalibrator()
        dq = cal.dequantize([1, 2, 3, 4], scale=0.5, zero_point=0.0)
        assert len(dq) == 4

    def test_dequantize_formula(self):
        cal = GPTQCalibrator()
        # (q - zp) * scale
        dq = cal.dequantize([4], scale=0.25, zero_point=0.0)
        assert abs(dq[0] - 1.0) < 1e-9

    def test_round_trip_close_to_original(self):
        cal = GPTQCalibrator()
        # Use only non-negative weights for zero_point=0 symmetric scheme
        w = [0.1, 0.5, 0.3, 0.9, 0.0]
        scale, zp = cal.compute_scale(w, bits=4)
        q = cal.quantize_weight(w, scale, zp, bits=4)
        dq = cal.dequantize(q, scale, zp)
        for orig, rec in zip(w, dq):
            assert abs(orig - rec) <= scale + 1e-6

    def test_dequantize_with_nonzero_zero_point(self):
        cal = GPTQCalibrator()
        # (q - zp) * scale: q=8, zp=8, scale=1.0 -> 0.0
        dq = cal.dequantize([8], scale=1.0, zero_point=8.0)
        assert abs(dq[0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_scale
# ---------------------------------------------------------------------------


class TestComputeScale:
    def test_scale_positive(self):
        cal = GPTQCalibrator()
        scale, _ = cal.compute_scale([1.0, -2.0, 0.5], bits=4)
        assert scale > 0

    def test_zero_point_is_zero(self):
        cal = GPTQCalibrator()
        _, zp = cal.compute_scale([1.0, -2.0, 0.5], bits=4)
        assert zp == 0.0

    def test_returns_tuple_of_two(self):
        cal = GPTQCalibrator()
        result = cal.compute_scale([1.0], bits=4)
        assert len(result) == 2

    def test_symmetric_scale_formula(self):
        cal = GPTQCalibrator()
        w = [3.0, -3.0, 1.5]
        scale, zp = cal.compute_scale(w, bits=4)
        # max_abs=3.0, max_q=2^(4-1)-1=7, scale=3/7
        expected = 3.0 / 7
        assert abs(scale - expected) < 1e-9
        assert zp == 0.0

    def test_all_zero_weights_scale_is_eps(self):
        cal = GPTQCalibrator()
        scale, _ = cal.compute_scale([0.0, 0.0, 0.0], bits=4)
        assert scale >= 1e-8

    def test_empty_weight_row(self):
        cal = GPTQCalibrator()
        scale, zp = cal.compute_scale([], bits=4)
        assert scale > 0
        assert zp == 0.0

    def test_scale_8bit(self):
        cal = GPTQCalibrator()
        scale, zp = cal.compute_scale([127.0], bits=8)
        # max_q = 2^7 - 1 = 127, scale = 127/127 = 1.0
        assert abs(scale - 1.0) < 1e-9
        assert zp == 0.0


# ---------------------------------------------------------------------------
# Round-trip reconstruction error
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_reconstruction_within_one_step_4bit(self):
        cal = GPTQCalibrator()
        # Use only non-negative weights: symmetric scheme maps negatives to 0
        w = [0.0, 0.25, 0.5, 0.75, 1.0]
        scale, zp = cal.compute_scale(w, bits=4)
        q = cal.quantize_weight(w, scale, zp, bits=4)
        dq = cal.dequantize(q, scale, zp)
        for orig, rec in zip(w, dq):
            assert abs(orig - rec) <= scale + 1e-6, (
                f"Reconstruction error {abs(orig - rec):.4f} > scale {scale:.4f} for w={orig}"
            )

    def test_reconstruction_within_one_step_8bit(self):
        cal = GPTQCalibrator()
        # Use only non-negative weights: symmetric scheme maps negatives to 0
        w = [0.0, 0.25, 0.5, 0.75, 1.0]
        scale, zp = cal.compute_scale(w, bits=8)
        q = cal.quantize_weight(w, scale, zp, bits=8)
        dq = cal.dequantize(q, scale, zp)
        for orig, rec in zip(w, dq):
            assert abs(orig - rec) <= scale + 1e-6

    def test_reconstruction_within_one_step_2bit(self):
        cal = GPTQCalibrator()
        # Use only non-negative weights: symmetric scheme maps negatives to 0
        w = [0.0, 0.5, 1.0]
        scale, zp = cal.compute_scale(w, bits=2)
        q = cal.quantize_weight(w, scale, zp, bits=2)
        dq = cal.dequantize(q, scale, zp)
        for orig, rec in zip(w, dq):
            assert abs(orig - rec) <= scale + 1e-6
