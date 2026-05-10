"""Tests for adaptive_speculative.py — Nightjar-inspired adaptive speculation."""

from __future__ import annotations

import pytest

from src.inference.adaptive_speculative import (
    AdaptiveSpecConfig,
    AdaptiveSpecController,
    AdaptiveSpeculativeDecoder,
)


class TestAdaptiveSpecController:
    """Nightjar-style EMA-based acceptance rate tracker."""

    def test_initial_K(self):
        ctrl = AdaptiveSpecController(AdaptiveSpecConfig(init_K=5))
        assert ctrl.K == 5

    def test_K_adjusts_up_when_acceptance_high(self):
        ctrl = AdaptiveSpecController(
            AdaptiveSpecConfig(
                init_K=3,
                target_acceptance=0.7,
                adapt_rate=1.0,
                ema_alpha=1.0,
            )
        )
        # High acceptance: 3/3 = 1.0. error = 1.0 - 0.7 = 0.3. delta = 1.0 * 0.3 * 3 = 0.9 → round = 1
        new_k = ctrl.update(3, 3)
        assert new_k == 4

    def test_K_adjusts_down_when_acceptance_low(self):
        ctrl = AdaptiveSpecController(
            AdaptiveSpecConfig(
                init_K=4,
                target_acceptance=0.7,
                adapt_rate=1.0,
                ema_alpha=1.0,
            )
        )
        # Low acceptance: 1/4 = 0.25. error = 0.25 - 0.7 = -0.45. delta = -1.8 → round = -2
        new_k = ctrl.update(1, 4)
        assert new_k == 2

    def test_K_clamped_to_min(self):
        ctrl = AdaptiveSpecController(
            AdaptiveSpecConfig(min_K=2, init_K=3, adapt_rate=10.0, ema_alpha=1.0)
        )
        new_k = ctrl.update(0, 4)
        assert new_k == 2

    def test_K_clamped_to_max(self):
        ctrl = AdaptiveSpecController(
            AdaptiveSpecConfig(max_K=6, init_K=4, adapt_rate=10.0, ema_alpha=1.0)
        )
        new_k = ctrl.update(4, 4)
        assert new_k == 6

    def test_ema_smoothing(self):
        ctrl = AdaptiveSpecController(
            AdaptiveSpecConfig(init_K=4, ema_alpha=0.5, adapt_rate=1.0)
        )
        ctrl.update(3, 4)  # 0.75
        ema_after_one = ctrl.ema_acceptance
        ctrl.update(1, 4)  # 0.25
        ema_after_two = ctrl.ema_acceptance
        # ema = alpha * round_rate + (1-alpha) * prev_ema
        expected_after_one = 0.5 * 0.75 + 0.5 * 0.7  # 0.7 is init
        assert abs(ema_after_one - expected_after_one) < 0.01

    def test_overall_acceptance_tracked(self):
        ctrl = AdaptiveSpecController(AdaptiveSpecConfig())
        ctrl.update(3, 4)
        ctrl.update(2, 4)
        ctrl.update(4, 4)
        assert abs(ctrl.overall_acceptance - 9 / 12) < 0.001

    def test_reset(self):
        ctrl = AdaptiveSpecController(AdaptiveSpecConfig(init_K=5))
        ctrl.update(1, 4)
        ctrl.reset()
        assert ctrl.K == 5
        assert ctrl.ema_acceptance == 0.7
        assert ctrl.overall_acceptance == 0.0

    def test_zero_proposed_returns_current_K(self):
        ctrl = AdaptiveSpecController(AdaptiveSpecConfig(init_K=3))
        new_k = ctrl.update(0, 0)
        assert new_k == 3

    def test_proposed_zero_not_divide_by_zero(self):
        ctrl = AdaptiveSpecController(AdaptiveSpecConfig())
        ctrl.update(0, 0)
        assert ctrl.ema_acceptance == 0.7  # unchanged


class TestAdaptiveSpecConfig:
    """Config dataclass validation."""

    def test_defaults(self):
        cfg = AdaptiveSpecConfig()
        assert cfg.min_K == 1
        assert cfg.max_K == 8
        assert cfg.init_K == 4
        assert cfg.target_acceptance == 0.7
        assert cfg.adapt_rate == 0.1
        assert cfg.ema_alpha == 0.3
