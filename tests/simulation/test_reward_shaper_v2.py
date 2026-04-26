"""Tests for reward_shaper_v2 — advanced reward shaping utilities."""
from __future__ import annotations

import torch

from src.simulation.reward_shaper_v2 import RewardShaperV2, shaped_reward, potential_based_reward


class TestShapedReward:
    def test_positive_reward_shaped(self):
        r = shaped_reward(raw=10.0, prev=5.0, scale=1.0)
        assert r > 0

    def test_negative_reward_shaped(self):
        r = shaped_reward(raw=5.0, prev=10.0, scale=1.0)
        assert r < 0

    def test_zero_scale_returns_zero(self):
        r = shaped_reward(raw=10.0, prev=5.0, scale=0.0)
        assert r == 0.0


class TestPotentialBasedReward:
    def test_higher_potential_positive(self):
        r = potential_based_reward(raw=0.0, new_potential=10.0, old_potential=5.0, gamma=0.9)
        assert r > 0

    def test_lower_potential_negative(self):
        r = potential_based_reward(raw=0.0, new_potential=3.0, old_potential=8.0, gamma=0.9)
        assert r < 0

    def test_gamma_scales(self):
        r1 = potential_based_reward(0.0, 10.0, 5.0, 0.9)
        r2 = potential_based_reward(0.0, 10.0, 5.0, 0.5)
        assert r1 != r2


class TestRewardShaperV2:
    def test_shapes_single_step(self):
        shaper = RewardShaperV2()
        shaped = shaper.shape(raw=5.0, state=torch.tensor([0.5, 0.5]))
        assert isinstance(shaped, float)

    def test_potential_function_changes(self):
        shaper = RewardShaperV2()
        r1 = shaper.shape(raw=0.0, state=torch.tensor([0.1, 0.2]))
        r2 = shaper.shape(raw=0.0, state=torch.tensor([0.9, 0.8]))
        assert r1 != r2

    def test_reset_clears_potential(self):
        shaper = RewardShaperV2()
        shaper.shape(raw=5.0, state=torch.tensor([1.0, 1.0]))
        shaper.reset()
        r = shaper.shape(raw=5.0, state=torch.tensor([1.0, 1.0]))
        assert r > 0  # first call after reset should give positive shaping
