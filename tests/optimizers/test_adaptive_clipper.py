"""Tests for adaptive gradient clipper."""

from __future__ import annotations

from src.optimizers.adaptive_clipper import AdaptiveGradientClipper


class TestAdaptiveGradientClipper:
    def test_clip_returns_float(self):
        clipper = AdaptiveGradientClipper()
        assert isinstance(clipper.clip(1.0), float)

    def test_grows_threshold_when_norms_high(self):
        clipper = AdaptiveGradientClipper(initial_threshold=1.0)
        for _ in range(10):
            clipper.clip(5.0)
        assert clipper.clip(5.0) > 1.0

    def test_decays_threshold_when_norms_low(self):
        clipper = AdaptiveGradientClipper(initial_threshold=1.0)
        for _ in range(10):
            clipper.clip(0.1)
        assert clipper.clip(0.1) <= 1.0
