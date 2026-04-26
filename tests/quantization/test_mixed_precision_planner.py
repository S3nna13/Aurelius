"""Tests for the mixed-precision planner module."""

from __future__ import annotations

from src.quantization.mixed_precision_planner import (
    LayerSensitivity,
    MixedPrecisionPlan,
    MixedPrecisionPlanner,
)

# ---------------------------------------------------------------------------
# LayerSensitivity fields
# ---------------------------------------------------------------------------


class TestLayerSensitivityFields:
    def test_has_layer_name(self):
        s = LayerSensitivity(layer_name="fc1", sensitivity_score=0.5, recommended_bits=4)
        assert s.layer_name == "fc1"

    def test_has_sensitivity_score(self):
        s = LayerSensitivity(layer_name="fc1", sensitivity_score=1.5, recommended_bits=8)
        assert s.sensitivity_score == 1.5

    def test_has_recommended_bits(self):
        s = LayerSensitivity(layer_name="fc1", sensitivity_score=0.5, recommended_bits=4)
        assert s.recommended_bits == 4

    def test_current_bits_default(self):
        s = LayerSensitivity(layer_name="fc1", sensitivity_score=0.5, recommended_bits=4)
        assert s.current_bits == 16

    def test_custom_current_bits(self):
        s = LayerSensitivity(
            layer_name="fc1", sensitivity_score=0.5, recommended_bits=4, current_bits=8
        )
        assert s.current_bits == 8


# ---------------------------------------------------------------------------
# MixedPrecisionPlan fields
# ---------------------------------------------------------------------------


class TestMixedPrecisionPlanFields:
    def _make_plan(self):
        return MixedPrecisionPlan(
            layer_assignments={"fc1": 4, "fc2": 8},
            total_bits_saved=100,
            compression_ratio=2.0,
        )

    def test_has_layer_assignments(self):
        p = self._make_plan()
        assert isinstance(p.layer_assignments, dict)

    def test_has_total_bits_saved(self):
        p = self._make_plan()
        assert p.total_bits_saved == 100

    def test_has_compression_ratio(self):
        p = self._make_plan()
        assert p.compression_ratio == 2.0

    def test_layer_assignments_values(self):
        p = self._make_plan()
        assert p.layer_assignments["fc1"] == 4
        assert p.layer_assignments["fc2"] == 8


# ---------------------------------------------------------------------------
# MixedPrecisionPlanner.score_sensitivity
# ---------------------------------------------------------------------------


class TestScoreSensitivity:
    def test_high_gradient_weight_ratio_gives_8_bits(self):
        planner = MixedPrecisionPlanner()
        # score > 1.0 -> 8 bits
        result = planner.score_sensitivity("layer", weight_magnitude=0.1, gradient_magnitude=10.0)
        assert result.recommended_bits == 8

    def test_medium_ratio_gives_4_bits(self):
        planner = MixedPrecisionPlanner()
        # score = 0.5/1.0 = 0.5 (between 0.1 and 1.0) -> 4 bits
        result = planner.score_sensitivity("layer", weight_magnitude=1.0, gradient_magnitude=0.5)
        assert result.recommended_bits == 4

    def test_low_ratio_gives_2_bits(self):
        planner = MixedPrecisionPlanner()
        # score = 0.01 / (1.0 + 1e-8) ≈ 0.01 < 0.1 -> 2 bits
        result = planner.score_sensitivity("layer", weight_magnitude=1.0, gradient_magnitude=0.01)
        assert result.recommended_bits == 2

    def test_sensitivity_score_formula(self):
        planner = MixedPrecisionPlanner()
        w, g = 2.0, 4.0
        result = planner.score_sensitivity("layer", weight_magnitude=w, gradient_magnitude=g)
        expected_score = g / (w + 1e-8)
        assert abs(result.sensitivity_score - expected_score) < 1e-6

    def test_zero_weight_magnitude(self):
        planner = MixedPrecisionPlanner()
        result = planner.score_sensitivity("layer", weight_magnitude=0.0, gradient_magnitude=1.0)
        # score = 1.0 / 1e-8 >> 1.0 -> 8 bits
        assert result.recommended_bits == 8

    def test_returns_layer_sensitivity(self):
        planner = MixedPrecisionPlanner()
        result = planner.score_sensitivity("my_layer", weight_magnitude=1.0, gradient_magnitude=0.5)
        assert isinstance(result, LayerSensitivity)

    def test_layer_name_preserved(self):
        planner = MixedPrecisionPlanner()
        result = planner.score_sensitivity(
            "test_layer", weight_magnitude=1.0, gradient_magnitude=0.5
        )
        assert result.layer_name == "test_layer"

    def test_exactly_at_1_boundary(self):
        planner = MixedPrecisionPlanner()
        # score = 1.0 -> NOT > 1.0, check next: > 0.1 -> 4 bits
        result = planner.score_sensitivity("layer", weight_magnitude=1.0, gradient_magnitude=1.0)
        assert result.recommended_bits == 4

    def test_exactly_at_0_1_boundary(self):
        planner = MixedPrecisionPlanner()
        # score = 0.1 -> NOT > 0.1, -> 2 bits
        result = planner.score_sensitivity("layer", weight_magnitude=1.0, gradient_magnitude=0.1)
        assert result.recommended_bits == 2


# ---------------------------------------------------------------------------
# MixedPrecisionPlanner.plan
# ---------------------------------------------------------------------------


class TestPlan:
    def _make_sensitivities(self):
        return [
            LayerSensitivity("fc1", sensitivity_score=2.0, recommended_bits=8),
            LayerSensitivity("fc2", sensitivity_score=0.5, recommended_bits=4),
            LayerSensitivity("fc3", sensitivity_score=0.05, recommended_bits=2),
        ]

    def _make_n_params(self):
        return {"fc1": 1000, "fc2": 2000, "fc3": 500}

    def test_returns_mixed_precision_plan(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        result = planner.plan(self._make_sensitivities(), self._make_n_params())
        assert isinstance(result, MixedPrecisionPlan)

    def test_layer_assignments_keys_match_input(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        sens = self._make_sensitivities()
        n_params = self._make_n_params()
        result = planner.plan(sens, n_params)
        assert set(result.layer_assignments.keys()) == {"fc1", "fc2", "fc3"}

    def test_all_assigned_bits_in_available(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0, available_bits=[2, 4, 8, 16])
        result = planner.plan(self._make_sensitivities(), self._make_n_params())
        for bits in result.layer_assignments.values():
            assert bits in [2, 4, 8, 16]

    def test_total_bits_saved_nonneg(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        result = planner.plan(self._make_sensitivities(), self._make_n_params())
        assert result.total_bits_saved >= 0

    def test_compression_ratio_at_least_1(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        result = planner.plan(self._make_sensitivities(), self._make_n_params())
        assert result.compression_ratio >= 1.0

    def test_empty_sensitivities(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        result = planner.plan([], {})
        assert isinstance(result, MixedPrecisionPlan)
        assert result.layer_assignments == {}

    def test_single_layer(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        sens = [LayerSensitivity("only_layer", sensitivity_score=0.5, recommended_bits=4)]
        n_params = {"only_layer": 1000}
        result = planner.plan(sens, n_params)
        assert "only_layer" in result.layer_assignments

    def test_compression_ratio_greater_when_low_bits(self):
        planner = MixedPrecisionPlanner(target_avg_bits=2.0, available_bits=[2, 4, 8, 16])
        sens = [
            LayerSensitivity("fc1", sensitivity_score=0.01, recommended_bits=2),
        ]
        n_params = {"fc1": 1000}
        result = planner.plan(sens, n_params)
        # 16 / 2 = 8
        assert result.compression_ratio >= 2.0

    def test_plan_respects_available_bits(self):
        planner = MixedPrecisionPlanner(target_avg_bits=8.0, available_bits=[4, 8])
        sens = [LayerSensitivity("fc1", sensitivity_score=2.0, recommended_bits=8)]
        n_params = {"fc1": 1000}
        result = planner.plan(sens, n_params)
        assert result.layer_assignments["fc1"] in [4, 8]

    def test_total_bits_saved_calculation(self):
        planner = MixedPrecisionPlanner(target_avg_bits=16.0, available_bits=[16])
        sens = [LayerSensitivity("fc1", sensitivity_score=0.5, recommended_bits=4, current_bits=16)]
        n_params = {"fc1": 100}
        result = planner.plan(sens, n_params)
        # assigned=16, saved = (16-16)*100 = 0
        assert result.total_bits_saved == 0


# ---------------------------------------------------------------------------
# MixedPrecisionPlanner.validate_plan
# ---------------------------------------------------------------------------


class TestValidatePlan:
    def test_true_when_within_budget(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 4, "fc2": 4},
            total_bits_saved=0,
            compression_ratio=4.0,
        )
        n_params = {"fc1": 100, "fc2": 100}
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is True

    def test_false_when_over_budget(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 16, "fc2": 16},
            total_bits_saved=0,
            compression_ratio=1.0,
        )
        n_params = {"fc1": 100, "fc2": 100}
        # avg_bits = 16, target = 4.0, 16 > 4.5 -> False
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is False

    def test_true_within_tolerance(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 4},
            total_bits_saved=0,
            compression_ratio=4.0,
        )
        n_params = {"fc1": 100}
        # avg = 4.0 <= 4.0 + 0.5 = 4.5 -> True
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is True

    def test_true_at_tolerance_boundary(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        # Mixed: 50 params at 4 bits + 50 params at 5 bits -> avg=4.5 -> True (4.5 <= 4.5)
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 4, "fc2": 5},
            total_bits_saved=0,
            compression_ratio=1.0,
        )
        n_params = {"fc1": 50, "fc2": 50}
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is True

    def test_empty_plan_returns_true(self):
        planner = MixedPrecisionPlanner()
        plan = MixedPrecisionPlan(layer_assignments={}, total_bits_saved=0, compression_ratio=1.0)
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params={}) is True

    def test_mixed_bits_within_budget(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 8, "fc2": 2},
            total_bits_saved=500,
            compression_ratio=3.2,
        )
        # avg = (8*100 + 2*100) / 200 = 1000/200 = 5.0 > 4.5 -> False
        n_params = {"fc1": 100, "fc2": 100}
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is False

    def test_validate_uses_weighted_average(self):
        planner = MixedPrecisionPlanner(target_avg_bits=4.0)
        # fc1 is tiny (1 param at 16 bits), fc2 is huge (999 params at 4 bits)
        # weighted avg ≈ (16 + 3996) / 1000 ≈ 4.012 -> within budget
        plan = MixedPrecisionPlan(
            layer_assignments={"fc1": 16, "fc2": 4},
            total_bits_saved=0,
            compression_ratio=1.0,
        )
        n_params = {"fc1": 1, "fc2": 999}
        assert planner.validate_plan(plan, target_avg_bits=4.0, n_params=n_params) is True
