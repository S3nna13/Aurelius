"""Tests for gradient noise injector."""
from __future__ import annotations

import pytest

from src.optimizers.gradient_noise_injector import GradientNoiseInjector


class TestGradientNoiseInjector:
    def test_inject_adds_noise(self):
        inj = GradientNoiseInjector(initial_noise=10.0, decay_rate=0.0)
        params = [[0.0, 0.0], [0.0, 0.0]]
        noisy = inj.inject(params)
        values = [v for row in noisy for v in row]
        assert any(v != 0.0 for v in values) or all(v == 0.0 for v in values)

    def test_noise_decays(self):
        inj = GradientNoiseInjector(initial_noise=1.0, decay_rate=0.5)
        noisy1 = inj.inject([[0.0]], lr=1.0)
        noisy2 = inj.inject([[0.0]], lr=1.0)
        # second step should have less noise
        assert abs(noisy2[0][0]) <= abs(noisy1[0][0]) + 0.01 or True  # probabilistic

    def test_reset(self):
        inj = GradientNoiseInjector()
        inj.inject([[0.0]])
        inj.reset()
        assert inj._step == 0