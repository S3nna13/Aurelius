"""Tests for the cosine warmup LR scheduler."""

from __future__ import annotations

import pytest
import torch
import torch.optim as optim

from src.training.cosine_warmup import (
    CosineWarmupSchedule,
    InverseSqrtScheduler,
    WarmupCosineScheduler,
    get_lr_multiplier,
)

# ------------------------------------------------------------------------------- #
# Helpers                                                                          #
# ------------------------------------------------------------------------------- #


def make_schedule(warmup=100, total=1000, min_lr_ratio=0.1):
    return CosineWarmupSchedule(warmup_steps=warmup, total_steps=total, min_lr_ratio=min_lr_ratio)


def make_optimizer(lr=1e-3):
    p = torch.nn.Parameter(torch.tensor(1.0))
    return optim.Adam([p], lr=lr), p


# ------------------------------------------------------------------------------- #
# Tests                                                                            #
# ------------------------------------------------------------------------------- #


def test_lr_at_step_0_is_zero():
    """At step 0 the multiplier should be 0 (start of warmup)."""
    schedule = make_schedule(warmup=100, total=1000)
    mult = get_lr_multiplier(0, schedule)
    assert mult == pytest.approx(0.0)


def test_lr_at_warmup_steps_is_one():
    """At the end of warmup the multiplier should be 1.0."""
    schedule = make_schedule(warmup=100, total=1000)
    mult = get_lr_multiplier(100, schedule)
    assert mult == pytest.approx(1.0, abs=1e-6)


def test_lr_at_total_steps_is_min_lr_ratio():
    """At total_steps the multiplier should equal min_lr_ratio."""
    min_ratio = 0.1
    schedule = make_schedule(warmup=100, total=1000, min_lr_ratio=min_ratio)
    mult = get_lr_multiplier(1000, schedule)
    assert mult == pytest.approx(min_ratio)


def test_lr_monotone_increases_during_warmup():
    """Multiplier should strictly increase during warmup."""
    schedule = make_schedule(warmup=50, total=500)
    mults = [get_lr_multiplier(s, schedule) for s in range(51)]
    for a, b in zip(mults, mults[1:]):
        assert b > a


def test_lr_monotone_decreases_after_warmup():
    """Multiplier should be non-increasing after warmup completes."""
    schedule = make_schedule(warmup=50, total=500)
    mults = [get_lr_multiplier(s, schedule) for s in range(50, 501)]
    for a, b in zip(mults, mults[1:]):
        assert b <= a + 1e-9  # allow tiny float noise


def test_min_lr_ratio_respected_at_end():
    """Multiplier after total_steps should never fall below min_lr_ratio."""
    min_ratio = 0.05
    schedule = make_schedule(warmup=10, total=100, min_lr_ratio=min_ratio)
    for step in [100, 200, 1000]:
        mult = get_lr_multiplier(step, schedule)
        assert mult >= min_ratio - 1e-9


def test_inverse_sqrt_shape():
    """InverseSqrtScheduler should decrease after warmup like 1/sqrt(t)."""
    opt, _ = make_optimizer(lr=1.0)
    warmup = 10
    scheduler = InverseSqrtScheduler(opt, warmup_steps=warmup)

    lrs = []
    for step in range(1, 50):
        scheduler.last_epoch = step
        lrs.append(scheduler.get_lr()[0])

    # After warmup, must be decreasing
    post_warmup = lrs[warmup:]
    for a, b in zip(post_warmup, post_warmup[1:]):
        assert b <= a + 1e-9


def test_warmup_cosine_scheduler_integrates_with_adam():
    """WarmupCosineScheduler should change optimizer lr without error."""
    opt, _ = make_optimizer(lr=1e-3)
    scheduler = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    for _ in range(20):
        opt.step()
        scheduler.step()
    # lr should be between 0 and base lr
    current_lr = opt.param_groups[0]["lr"]
    assert 0.0 <= current_lr <= 1e-3 + 1e-9


def test_multiple_param_groups():
    """Scheduler should correctly scale multiple param group lrs."""
    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(2.0))
    opt = optim.Adam([{"params": [p1], "lr": 1e-3}, {"params": [p2], "lr": 1e-2}])
    scheduler = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    # Step into cosine decay region
    for _ in range(50):
        opt.step()
        scheduler.step()
    lr1 = opt.param_groups[0]["lr"]
    lr2 = opt.param_groups[1]["lr"]
    # Second group lr should still be ~10x the first
    assert lr2 == pytest.approx(lr1 * 10, rel=1e-3)


def test_step_advances_correctly():
    """Calling scheduler.step() should advance the schedule consistently."""
    opt, _ = make_optimizer(lr=1.0)
    scheduler = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50)

    lrs = []
    for _ in range(60):
        opt.step()
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Should reach warmup peak around step 5
    peak = max(lrs[:10])
    assert peak == pytest.approx(1.0, abs=0.05)
    # Should eventually settle at min_lr_ratio * base_lr
    assert lrs[-1] == pytest.approx(0.1, abs=0.02)
