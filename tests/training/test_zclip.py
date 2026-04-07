"""Tests for ZClip adaptive gradient clipping."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.zclip import ZClip


def _make_param_with_grad(grad_value: float, size: int = 10) -> nn.Parameter:
    """Create a parameter with a fixed gradient."""
    p = nn.Parameter(torch.zeros(size))
    p.grad = torch.full((size,), grad_value / math.sqrt(size))
    return p


def _params_with_norm(target_norm: float, size: int = 100) -> list[nn.Parameter]:
    """Return a list of parameters whose gradient has the given L2 norm."""
    p = nn.Parameter(torch.zeros(size))
    # Each element = target_norm / sqrt(size) gives ||grad|| == target_norm
    p.grad = torch.full((size,), target_norm / math.sqrt(size))
    return [p]


def _compute_grad_norm(params: list[nn.Parameter]) -> float:
    """Compute L2 grad norm for a list of parameters."""
    total = sum(p.grad.float().norm() ** 2 for p in params if p.grad is not None)
    return float(total ** 0.5)


# ---------------------------------------------------------------------------
# test_zclip_warmup_clips_to_fallback
# ---------------------------------------------------------------------------

def test_zclip_warmup_clips_to_fallback():
    """During warmup, a huge gradient norm is clipped to <= fallback_clip * 1.01."""
    fallback = 1.0
    zclip = ZClip(params=[], fallback_clip=fallback, min_warmup_steps=100)

    # Inject a gradient with norm = 100 (way above fallback)
    params = _params_with_norm(100.0)
    pre_norm = zclip.clip_grad_norm_(params)

    assert pre_norm == pytest.approx(100.0, rel=1e-4), "Return value should be pre-clip norm"
    post_norm = _compute_grad_norm(params)
    assert post_norm <= fallback * 1.01, (
        f"Post-clip norm {post_norm:.4f} should be <= fallback {fallback}"
    )


# ---------------------------------------------------------------------------
# test_zclip_no_clip_normal_gradient
# ---------------------------------------------------------------------------

def test_zclip_no_clip_normal_gradient():
    """After warmup, a gradient at the running mean is NOT clipped."""
    warmup = 10
    zclip = ZClip(
        params=[],
        z_threshold=2.5,
        ema_alpha=0.1,
        min_warmup_steps=warmup,
        fallback_clip=1.0,
    )

    # Run through warmup with norm=1.0 so EMA converges near 1.0
    for _ in range(warmup):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    # Now inject a gradient at exactly the EMA mean (z-score ≈ 0)
    mean = zclip._ema_mean
    params = _params_with_norm(mean)
    pre_norm = zclip.clip_grad_norm_(params)
    post_norm = _compute_grad_norm(params)

    # Norm should be essentially unchanged (no clipping)
    assert post_norm == pytest.approx(pre_norm, rel=1e-3), (
        "Normal gradient (at mean) should not be clipped"
    )


# ---------------------------------------------------------------------------
# test_zclip_clips_spike
# ---------------------------------------------------------------------------

def test_zclip_clips_spike():
    """After warmup, a spike (mean + 10*std) is clipped down."""
    warmup = 20
    zclip = ZClip(
        params=[],
        z_threshold=2.5,
        ema_alpha=0.1,
        min_warmup_steps=warmup,
        fallback_clip=1.0,
    )

    # Warm up with consistent norm=1.0
    for _ in range(warmup):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    mean = zclip._ema_mean
    std = math.sqrt(zclip._ema_var + 1e-8)
    spike_norm = mean + 10 * std  # well above z_threshold=2.5

    params = _params_with_norm(spike_norm)
    pre_norm = zclip.clip_grad_norm_(params)
    post_norm = _compute_grad_norm(params)

    assert pre_norm == pytest.approx(spike_norm, rel=1e-3), (
        "Return value should be pre-clip norm"
    )
    assert post_norm < pre_norm, (
        f"Spike norm {pre_norm:.4f} should have been clipped; post={post_norm:.4f}"
    )
    # Clipped to mean + z_threshold * std
    expected_clip = mean + 2.5 * std
    assert post_norm == pytest.approx(expected_clip, rel=1e-3), (
        f"Post-clip norm {post_norm:.4f} should equal mean + z*std = {expected_clip:.4f}"
    )


# ---------------------------------------------------------------------------
# test_zclip_ema_updates
# ---------------------------------------------------------------------------

def test_zclip_ema_updates():
    """After several steps, _ema_mean is greater than 0 (EMA is updating)."""
    zclip = ZClip(params=[], ema_alpha=0.1, min_warmup_steps=5)

    for _ in range(10):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    assert zclip._ema_mean > 0.0, (
        f"_ema_mean should be > 0 after updates, got {zclip._ema_mean}"
    )
    assert zclip._step == 10, f"_step should be 10, got {zclip._step}"


# ---------------------------------------------------------------------------
# test_zclip_returns_prenorm
# ---------------------------------------------------------------------------

def test_zclip_returns_prenorm():
    """The return value is the pre-clip gradient norm, not the post-clip norm."""
    zclip = ZClip(params=[], fallback_clip=1.0, min_warmup_steps=100)

    target_norm = 50.0
    params = _params_with_norm(target_norm)
    returned = zclip.clip_grad_norm_(params)

    assert returned == pytest.approx(target_norm, rel=1e-3), (
        f"Return value {returned:.4f} should match pre-clip norm {target_norm}"
    )
    # Confirm clipping did happen (post-norm is smaller)
    post_norm = _compute_grad_norm(params)
    assert post_norm < target_norm, "Gradient should have been clipped"
