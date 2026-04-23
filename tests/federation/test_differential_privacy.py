"""Tests for src/federation/differential_privacy.py."""

from __future__ import annotations

import math
import pytest

from src.federation.differential_privacy import (
    DPMechanism,
    DifferentialPrivacy,
    DP_MECHANISM,
    PrivacyBudget,
)


# ---------------------------------------------------------------------------
# DPMechanism enum
# ---------------------------------------------------------------------------

class TestDPMechanism:
    def test_enum_count(self):
        assert len(DPMechanism) == 3

    def test_gaussian_value(self):
        assert DPMechanism.GAUSSIAN == "gaussian"

    def test_laplace_value(self):
        assert DPMechanism.LAPLACE == "laplace"

    def test_randomized_response_value(self):
        assert DPMechanism.RANDOMIZED_RESPONSE == "randomized_response"

    def test_str_subclass(self):
        assert isinstance(DPMechanism.GAUSSIAN, str)


# ---------------------------------------------------------------------------
# PrivacyBudget dataclass
# ---------------------------------------------------------------------------

class TestPrivacyBudget:
    def test_epsilon_stored(self):
        pb = PrivacyBudget(epsilon=1.0)
        assert pb.epsilon == 1.0

    def test_delta_default(self):
        pb = PrivacyBudget(epsilon=1.0)
        assert abs(pb.delta - 1e-5) < 1e-10

    def test_delta_explicit(self):
        pb = PrivacyBudget(epsilon=1.0, delta=1e-3)
        assert abs(pb.delta - 1e-3) < 1e-10

    def test_consumed_epsilon_default(self):
        pb = PrivacyBudget(epsilon=1.0)
        assert pb.consumed_epsilon == 0.0

    def test_consumed_epsilon_explicit(self):
        pb = PrivacyBudget(epsilon=2.0, consumed_epsilon=0.5)
        assert pb.consumed_epsilon == 0.5

    def test_fields_mutable(self):
        pb = PrivacyBudget(epsilon=1.0)
        pb.consumed_epsilon = 0.3
        assert pb.consumed_epsilon == 0.3


# ---------------------------------------------------------------------------
# DifferentialPrivacy – clip_gradient
# ---------------------------------------------------------------------------

class TestClipGradient:
    def test_clip_within_norm_unchanged(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        g = [0.3, 0.4]  # norm = 0.5 < 1.0
        result = dp.clip_gradient(g, max_norm=1.0)
        assert abs(result[0] - 0.3) < 1e-9
        assert abs(result[1] - 0.4) < 1e-9

    def test_clip_over_norm_scaled(self):
        dp = DifferentialPrivacy()
        g = [3.0, 4.0]  # norm = 5.0
        result = dp.clip_gradient(g, max_norm=1.0)
        norm_result = math.sqrt(sum(x ** 2 for x in result))
        assert abs(norm_result - 1.0) < 1e-9

    def test_clip_result_norm_le_max(self):
        dp = DifferentialPrivacy()
        g = [10.0, 10.0, 10.0]
        result = dp.clip_gradient(g, max_norm=2.0)
        norm_result = math.sqrt(sum(x ** 2 for x in result))
        assert norm_result <= 2.0 + 1e-9

    def test_clip_zero_vector(self):
        dp = DifferentialPrivacy()
        result = dp.clip_gradient([0.0, 0.0], max_norm=1.0)
        assert result == [0.0, 0.0]

    def test_clip_returns_list(self):
        dp = DifferentialPrivacy()
        result = dp.clip_gradient([1.0, 2.0])
        assert isinstance(result, list)

    def test_clip_length_preserved(self):
        dp = DifferentialPrivacy()
        g = [1.0, 2.0, 3.0, 4.0]
        result = dp.clip_gradient(g)
        assert len(result) == len(g)

    def test_clip_direction_preserved(self):
        dp = DifferentialPrivacy()
        g = [3.0, 4.0]  # norm 5
        result = dp.clip_gradient(g, max_norm=1.0)
        # Direction should be preserved (ratio is same)
        assert abs(result[0] / result[1] - 3.0 / 4.0) < 1e-9

    def test_clip_large_max_norm_no_change(self):
        dp = DifferentialPrivacy()
        g = [1.0, 1.0]
        result = dp.clip_gradient(g, max_norm=1000.0)
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# DifferentialPrivacy – add_gaussian_noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def test_output_length_same(self):
        dp = DifferentialPrivacy(seed=0)
        values = [1.0, 2.0, 3.0]
        result = dp.add_gaussian_noise(values)
        assert len(result) == 3

    def test_output_differs_from_input(self):
        dp = DifferentialPrivacy(seed=99)
        values = [1.0, 2.0, 3.0]
        result = dp.add_gaussian_noise(values)
        # At least one element should differ
        assert any(abs(r - v) > 1e-9 for r, v in zip(result, values))

    def test_returns_list(self):
        dp = DifferentialPrivacy()
        assert isinstance(dp.add_gaussian_noise([1.0]), list)

    def test_explicit_sigma(self):
        dp = DifferentialPrivacy(seed=1)
        values = [0.0] * 100
        result = dp.add_gaussian_noise(values, sigma=0.0)
        # With sigma=0, noise should be zero
        assert all(abs(r) < 1e-9 for r in result)

    def test_single_value(self):
        dp = DifferentialPrivacy(seed=2)
        result = dp.add_gaussian_noise([5.0])
        assert len(result) == 1

    def test_deterministic_with_same_seed(self):
        dp1 = DifferentialPrivacy(seed=42)
        dp2 = DifferentialPrivacy(seed=42)
        v = [1.0, 2.0, 3.0]
        assert dp1.add_gaussian_noise(v) == dp2.add_gaussian_noise(v)

    def test_different_seed_different_results(self):
        dp1 = DifferentialPrivacy(seed=1)
        dp2 = DifferentialPrivacy(seed=2)
        v = [1.0, 2.0, 3.0]
        r1 = dp1.add_gaussian_noise(v)
        r2 = dp2.add_gaussian_noise(v)
        assert r1 != r2


# ---------------------------------------------------------------------------
# DifferentialPrivacy – add_laplace_noise
# ---------------------------------------------------------------------------

class TestLaplaceNoise:
    def test_output_length_same(self):
        dp = DifferentialPrivacy(seed=0)
        values = [1.0, 2.0, 3.0]
        result = dp.add_laplace_noise(values)
        assert len(result) == 3

    def test_output_differs_from_input(self):
        dp = DifferentialPrivacy(seed=10)
        values = [1.0, 2.0, 3.0]
        result = dp.add_laplace_noise(values)
        assert any(abs(r - v) > 1e-9 for r, v in zip(result, values))

    def test_returns_list(self):
        dp = DifferentialPrivacy()
        assert isinstance(dp.add_laplace_noise([1.0]), list)

    def test_single_value(self):
        dp = DifferentialPrivacy(seed=3)
        result = dp.add_laplace_noise([5.0])
        assert len(result) == 1

    def test_deterministic_with_same_seed(self):
        dp1 = DifferentialPrivacy(seed=42)
        dp2 = DifferentialPrivacy(seed=42)
        v = [1.0, 2.0]
        assert dp1.add_laplace_noise(v) == dp2.add_laplace_noise(v)


# ---------------------------------------------------------------------------
# DifferentialPrivacy – privacy_budget
# ---------------------------------------------------------------------------

class TestPrivacyBudgetMethod:
    def test_returns_privacy_budget(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        pb = dp.privacy_budget()
        assert isinstance(pb, PrivacyBudget)

    def test_budget_epsilon_matches_init(self):
        dp = DifferentialPrivacy(epsilon=2.0)
        pb = dp.privacy_budget()
        assert abs(pb.epsilon - 2.0) < 1e-9

    def test_budget_delta_matches_init(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-3)
        pb = dp.privacy_budget()
        assert abs(pb.delta - 1e-3) < 1e-10

    def test_budget_initial_consumed_zero(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        pb = dp.privacy_budget()
        assert pb.consumed_epsilon == 0.0


# ---------------------------------------------------------------------------
# DifferentialPrivacy – consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_within_budget_returns_true(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        assert dp.consume(0.5) is True

    def test_exactly_at_budget_returns_true(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        assert dp.consume(1.0) is True

    def test_over_budget_returns_false(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        dp.consume(0.5)
        assert dp.consume(0.6) is False

    def test_consumed_accumulates(self):
        dp = DifferentialPrivacy(epsilon=2.0)
        dp.consume(0.5)
        dp.consume(0.5)
        pb = dp.privacy_budget()
        assert abs(pb.consumed_epsilon - 1.0) < 1e-9

    def test_false_once_exceeded(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        dp.consume(1.1)
        result = dp.consume(0.0)
        assert result is False

    def test_returns_bool(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        assert isinstance(dp.consume(0.1), bool)


# ---------------------------------------------------------------------------
# DP_MECHANISM singleton
# ---------------------------------------------------------------------------

class TestDPMechanismSingleton:
    def test_exists(self):
        assert DP_MECHANISM is not None

    def test_is_differential_privacy(self):
        assert isinstance(DP_MECHANISM, DifferentialPrivacy)

    def test_has_epsilon(self):
        assert hasattr(DP_MECHANISM, "epsilon")

    def test_has_delta(self):
        assert hasattr(DP_MECHANISM, "delta")
