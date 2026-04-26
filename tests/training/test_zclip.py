"""Tests for ZClip adaptive gradient clipping."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.zclip import ZClip


def _params_with_norm(target_norm: float, size: int = 100) -> list[nn.Parameter]:
    """Return a list of parameters whose gradient has the given L2 norm."""
    p = nn.Parameter(torch.zeros(size))
    p.grad = torch.full((size,), target_norm / math.sqrt(size))
    return [p]


def _compute_grad_norm(params: list[nn.Parameter]) -> float:
    """Compute L2 grad norm for a list of parameters."""
    total = sum(p.grad.float().norm() ** 2 for p in params if p.grad is not None)
    return float(total**0.5)


# ---------------------------------------------------------------------------
# test_zclip_warmup_clips_to_fallback
# ---------------------------------------------------------------------------


def test_zclip_warmup_clips_to_fallback():
    """During warmup, a huge gradient norm is clipped to <= fallback_clip * 1.01."""
    fallback = 1.0
    zclip = ZClip(fallback_clip=fallback, min_warmup_steps=100)

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
    """After warmup, a gradient at the running mean is NOT clipped.

    Uses ema_alpha=0.5 with 20 warmup steps of norm=1.0 so _ema_mean converges
    close to 1.0 before the test assertion. Verifies the gradient is unchanged
    (not merely that z-score of the mean is 0).
    """
    warmup = 20
    zclip = ZClip(z_threshold=2.5, ema_alpha=0.5, min_warmup_steps=warmup, fallback_clip=1.0)

    # Converge the EMA with consistent norm=1.0
    for _ in range(warmup):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    # EMA mean should be meaningfully close to 1.0 with alpha=0.5 and 20 steps
    assert zclip._ema_mean == pytest.approx(1.0, abs=0.1), (
        f"EMA mean should have converged near 1.0, got {zclip._ema_mean:.4f}"
    )

    # Inject gradient at exactly the EMA mean — z-score == 0, must not be clipped
    mean = zclip._ema_mean
    params = _params_with_norm(mean)
    pre_norm = zclip.clip_grad_norm_(params)
    post_norm = _compute_grad_norm(params)

    assert post_norm == pytest.approx(pre_norm, rel=1e-3), (
        "Normal gradient (at mean) should not be clipped"
    )


# ---------------------------------------------------------------------------
# test_zclip_clips_spike
# ---------------------------------------------------------------------------


def test_zclip_clips_spike():
    """After warmup, a spike (mean + 10*std) is clipped down to mean + z_threshold*std."""
    warmup = 20
    zclip = ZClip(z_threshold=2.5, ema_alpha=0.1, min_warmup_steps=warmup, fallback_clip=1.0)

    for _ in range(warmup):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    mean = zclip._ema_mean
    std = math.sqrt(zclip._ema_var + 1e-8)
    spike_norm = mean + 10 * std  # well above z_threshold=2.5

    params = _params_with_norm(spike_norm)
    pre_norm = zclip.clip_grad_norm_(params)
    post_norm = _compute_grad_norm(params)

    assert pre_norm == pytest.approx(spike_norm, rel=1e-3), "Return value should be pre-clip norm"
    assert post_norm < pre_norm, f"Spike norm {pre_norm:.4f} should have been clipped"
    expected_clip = mean + 2.5 * std
    assert post_norm == pytest.approx(expected_clip, rel=1e-3), (
        f"Post-clip norm {post_norm:.4f} should equal mean + z*std = {expected_clip:.4f}"
    )


# ---------------------------------------------------------------------------
# test_zclip_ema_updates
# ---------------------------------------------------------------------------


def test_zclip_ema_updates():
    """After several steps, _ema_mean is > 0 and _step is incremented."""
    zclip = ZClip(ema_alpha=0.1, min_warmup_steps=5)

    for _ in range(10):
        ps = _params_with_norm(1.0)
        zclip.clip_grad_norm_(ps)

    assert zclip._ema_mean > 0.0, f"_ema_mean should be > 0 after updates, got {zclip._ema_mean}"
    assert zclip._step == 10, f"_step should be 10, got {zclip._step}"


# ---------------------------------------------------------------------------
# test_zclip_returns_prenorm
# ---------------------------------------------------------------------------


def test_zclip_returns_prenorm():
    """The return value is the pre-clip norm, not the post-clip norm."""
    zclip = ZClip(fallback_clip=1.0, min_warmup_steps=100)

    target_norm = 50.0
    params = _params_with_norm(target_norm)
    returned = zclip.clip_grad_norm_(params)

    assert returned == pytest.approx(target_norm, rel=1e-3), (
        f"Return value {returned:.4f} should match pre-clip norm {target_norm}"
    )
    post_norm = _compute_grad_norm(params)
    assert post_norm < target_norm, "Gradient should have been clipped"


# ---------------------------------------------------------------------------
# test_zclip_empty_params
# ---------------------------------------------------------------------------


def test_zclip_empty_params():
    """Returns 0.0 and does not update EMA state when no params have grads."""
    zclip = ZClip()

    result = zclip.clip_grad_norm_([])
    assert result == 0.0
    assert zclip._step == 0, "_step must not increment for empty params"
    assert zclip._ema_mean == 0.0, "_ema_mean must not change for empty params"

    # Also test a param list where no grads are set
    p = nn.Parameter(torch.zeros(10))  # no .grad
    result2 = zclip.clip_grad_norm_([p])
    assert result2 == 0.0
    assert zclip._step == 0
