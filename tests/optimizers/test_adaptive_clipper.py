"""Tests for adaptive gradient clipper."""

from __future__ import annotations

import math

import torch

from src.optimizers.adaptive_clipper import AdaGC, AdaptiveGradientClipper, ZClip


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


class TestZClip:
    def test_clip_returns_float(self):
        zclip = ZClip()
        assert isinstance(zclip.clip(1.0), float)

    def test_no_clip_for_typical_norm(self):
        zclip = ZClip(z_threshold=3.0, beta=0.9)
        # Seed EMA with moderate norms
        for _ in range(20):
            zclip.clip(1.0)
        result = zclip.clip(1.0)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_clips_outlier_norm(self):
        zclip = ZClip(z_threshold=3.0, beta=0.9)
        for _ in range(20):
            zclip.clip(1.0)
        outlier = 100.0
        clipped = zclip.clip(outlier)
        assert clipped < outlier
        assert clipped > 0.0

    def test_ema_alpha_alias(self):
        zclip = ZClip(ema_alpha=0.01)
        assert zclip.beta == 0.99

    def test_beta_and_ema_alpha_conflict(self):
        try:
            ZClip(beta=0.95, ema_alpha=0.05)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_clip_grad_norm__clips_parameters(self):
        zclip = ZClip(z_threshold=3.0, beta=0.9)
        p = torch.nn.Parameter(torch.ones(10))
        p.grad = torch.ones(10) * 10.0
        # Seed EMA with smaller norms first
        for _ in range(20):
            zclip.clip(1.0)
        pre_norm = float(p.grad.norm())
        zclip.clip_grad_norm_([p])
        post_norm = float(p.grad.norm())
        assert post_norm < pre_norm

    def test_clip_grad_norm__returns_pre_clip_norm(self):
        zclip = ZClip()
        p = torch.nn.Parameter(torch.ones(5))
        p.grad = torch.ones(5) * 2.0
        returned = zclip.clip_grad_norm_([p])
        expected_norm = math.sqrt(5 * 4.0)
        assert math.isclose(returned, expected_norm, rel_tol=1e-4)

    def test_warmup_fallback_clip(self):
        zclip = ZClip(warmup_steps=5, fallback_clip=1.0)
        assert zclip.clip(2.0) == 1.0
        # After warmup, normal z-score logic applies
        for _ in range(5):
            zclip.clip(1.0)
        result = zclip.clip(1.0)
        assert result == 1.0

    def test_handles_nan_norm(self):
        zclip = ZClip()
        assert math.isnan(zclip.clip(float("nan")))

    def test_handles_empty_params(self):
        zclip = ZClip()
        assert zclip.clip_grad_norm_([]) == 0.0


class TestAdaGC:
    def test_clip_grads_returns_dict(self):
        adagc = AdaGC()
        p = torch.nn.Parameter(torch.ones(5))
        p.grad = torch.ones(5)
        norms = adagc.clip_grads_([p])
        assert isinstance(norms, dict)
        assert id(p) in norms

    def test_per_tensor_clipping(self):
        adagc = AdaGC(z_threshold=3.0, beta=0.9)
        p1 = torch.nn.Parameter(torch.ones(5))
        p2 = torch.nn.Parameter(torch.ones(5))
        # Seed EMAs
        for _ in range(10):
            p1.grad = torch.ones(5) * 0.1
            p2.grad = torch.ones(5) * 0.1
            adagc.clip_grads_([p1, p2])

        # Now give p1 an outlier, p2 stays normal
        p1.grad = torch.ones(5) * 100.0
        p2.grad = torch.ones(5) * 0.1
        pre_norm_p1 = float(p1.grad.norm())
        adagc.clip_grads_([p1, p2])
        post_norm_p1 = float(p1.grad.norm())
        post_norm_p2 = float(p2.grad.norm())

        assert post_norm_p1 < pre_norm_p1
        assert math.isclose(post_norm_p2, 0.1 * math.sqrt(5), rel_tol=1e-4)

    def test_per_tensor_state_isolated(self):
        adagc = AdaGC(beta=0.9)
        p = torch.nn.Parameter(torch.ones(3))
        p.grad = torch.ones(3) * 2.0
        adagc.clip_grads_([p])
        tid = id(p)
        assert tid in adagc._state
        ema, ema_sq, step = adagc._state[tid]
        expected_norm = 2.0 * math.sqrt(3)
        assert step == 1
        assert math.isclose(ema, expected_norm, rel_tol=1e-6)
        assert math.isclose(ema_sq, expected_norm**2, rel_tol=1e-6)
