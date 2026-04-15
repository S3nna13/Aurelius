"""
Tests for src/training/lr_schedulers.py

Pure PyTorch only. Tiny configs are used throughout.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.lr_schedulers import (
    SchedulerConfig,
    cosine_schedule_with_warmup,
    linear_schedule_with_warmup,
    polynomial_schedule,
    wsd_schedule,
    WarmupCosineScheduler,
    CyclicCosineScheduler,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def tiny_model_and_optimizer(lr: float = 1.0):
    """Return a tiny nn.Linear and SGD optimizer."""
    model = nn.Linear(4, 4, bias=False)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


def tiny_config():
    return SchedulerConfig(warmup_steps=10, total_steps=100, min_lr_ratio=0.1, num_cycles=0.5)


# ---------------------------------------------------------------------------
# 1. SchedulerConfig defaults
# ---------------------------------------------------------------------------

def test_scheduler_config_defaults():
    cfg = SchedulerConfig()
    assert cfg.warmup_steps == 1000
    assert cfg.total_steps == 10000
    assert cfg.min_lr_ratio == 0.1
    assert cfg.num_cycles == 0.5


# ---------------------------------------------------------------------------
# 2. cosine_schedule_with_warmup — step 0 ≈ 0
# ---------------------------------------------------------------------------

def test_cosine_step0_near_zero():
    cfg = tiny_config()
    val = cosine_schedule_with_warmup(0, cfg)
    assert val == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. cosine_schedule_with_warmup — at warmup_steps = 1.0
# ---------------------------------------------------------------------------

def test_cosine_at_warmup_equals_one():
    cfg = tiny_config()
    val = cosine_schedule_with_warmup(cfg.warmup_steps, cfg)
    assert val == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. cosine_schedule_with_warmup — at total_steps = min_lr_ratio
# ---------------------------------------------------------------------------

def test_cosine_at_total_steps_equals_min_lr():
    cfg = tiny_config()
    val = cosine_schedule_with_warmup(cfg.total_steps, cfg)
    assert val == pytest.approx(cfg.min_lr_ratio, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. linear_schedule — step 0 ≈ 0
# ---------------------------------------------------------------------------

def test_linear_step0_near_zero():
    cfg = tiny_config()
    val = linear_schedule_with_warmup(0, cfg)
    assert val == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. linear_schedule — at warmup_steps = 1.0
# ---------------------------------------------------------------------------

def test_linear_at_warmup_equals_one():
    cfg = tiny_config()
    val = linear_schedule_with_warmup(cfg.warmup_steps, cfg)
    assert val == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. polynomial_schedule power=1 is linear in (remaining / decay_steps)
# ---------------------------------------------------------------------------

def test_polynomial_power1_matches_linear_decay():
    cfg = tiny_config()
    # polynomial with power=1: ((total - step) / (total - warmup)) ^ 1
    # This is a linear ramp from 1.0 at warmup to 0.0 at total (before min_ratio clamp).
    # Verify the formula directly.
    step = 50  # midway in decay phase
    poly_val = polynomial_schedule(step, cfg, power=1.0)
    expected_raw = (cfg.total_steps - step) / (cfg.total_steps - cfg.warmup_steps)
    expected = max(cfg.min_lr_ratio, min(1.0, expected_raw))
    assert poly_val == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. wsd_schedule — warmup region grows
# ---------------------------------------------------------------------------

def test_wsd_warmup_region_monotone():
    steps_before = 5
    steps_after = 8
    v1 = wsd_schedule(steps_before, warmup_steps=10, stable_steps=10, decay_steps=10)
    v2 = wsd_schedule(steps_after, warmup_steps=10, stable_steps=10, decay_steps=10)
    assert v2 > v1


# ---------------------------------------------------------------------------
# 9. wsd_schedule — stable region = 1.0
# ---------------------------------------------------------------------------

def test_wsd_stable_region_is_one():
    # stable region: steps in [10, 20)
    for step in [10, 15, 19]:
        val = wsd_schedule(step, warmup_steps=10, stable_steps=10, decay_steps=10)
        assert val == pytest.approx(1.0, abs=1e-9), f"Failed at step {step}"


# ---------------------------------------------------------------------------
# 10. wsd_schedule — decay region decreases
# ---------------------------------------------------------------------------

def test_wsd_decay_region_decreases():
    # decay region: steps in [20, 30)
    v1 = wsd_schedule(21, warmup_steps=10, stable_steps=10, decay_steps=10, min_lr_ratio=0.1)
    v2 = wsd_schedule(28, warmup_steps=10, stable_steps=10, decay_steps=10, min_lr_ratio=0.1)
    assert v2 < v1


# ---------------------------------------------------------------------------
# 11. WarmupCosineScheduler.step() actually changes optimizer lr
# ---------------------------------------------------------------------------

def test_warmup_cosine_scheduler_step_changes_lr():
    cfg = tiny_config()
    _, optimizer = tiny_model_and_optimizer(lr=1.0)
    scheduler = WarmupCosineScheduler(optimizer, cfg)

    lrs = []
    for _ in range(5):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Should not all be the same during warmup
    assert len(set(round(v, 8) for v in lrs)) > 1


# ---------------------------------------------------------------------------
# 12. Scheduler lr stays in [min_lr_ratio, 1.0] after warmup completes
# ---------------------------------------------------------------------------

def test_scheduler_lr_bounded():
    cfg = tiny_config()
    # After warmup, multiplier must remain in [min_lr_ratio, 1.0].
    # During warmup (step < warmup_steps) the value ramps from 0 → 1.
    for step in range(cfg.warmup_steps, cfg.total_steps + 5):
        multiplier = cosine_schedule_with_warmup(step, cfg)
        assert multiplier >= cfg.min_lr_ratio - 1e-9, (
            f"multiplier={multiplier} below min_lr_ratio at step {step}"
        )
        assert multiplier <= 1.0 + 1e-9, (
            f"multiplier={multiplier} above 1.0 at step {step}"
        )


# ---------------------------------------------------------------------------
# 13. CyclicCosineScheduler.get_lr returns float
# ---------------------------------------------------------------------------

def test_cyclic_cosine_returns_float():
    cfg = tiny_config()
    sched = CyclicCosineScheduler(cfg)
    for step in [0, 5, 10, 50, 99, 100]:
        val = sched.get_lr(step)
        assert isinstance(val, float), f"Expected float, got {type(val)} at step {step}"
        assert cfg.min_lr_ratio - 1e-9 <= val <= 1.0 + 1e-9, f"Out of range at step {step}: {val}"


# ---------------------------------------------------------------------------
# 14. get_scheduler "cosine" returns a scheduler
# ---------------------------------------------------------------------------

def test_get_scheduler_cosine_returns_scheduler():
    from torch.optim.lr_scheduler import LambdaLR
    cfg = tiny_config()
    _, optimizer = tiny_model_and_optimizer()
    sched = get_scheduler("cosine", optimizer, cfg)
    assert isinstance(sched, LambdaLR)


# ---------------------------------------------------------------------------
# 15. Schedule is monotonically increasing during warmup
# ---------------------------------------------------------------------------

def test_cosine_monotone_increase_during_warmup():
    cfg = tiny_config()
    vals = [cosine_schedule_with_warmup(s, cfg) for s in range(cfg.warmup_steps + 1)]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1], f"Not monotone at step {i}: {vals[i-1]} -> {vals[i]}"


# ---------------------------------------------------------------------------
# 16. Cosine schedule is monotonically decreasing after warmup
# ---------------------------------------------------------------------------

def test_cosine_monotone_decrease_after_warmup():
    cfg = tiny_config()
    vals = [
        cosine_schedule_with_warmup(s, cfg)
        for s in range(cfg.warmup_steps, cfg.total_steps + 1)
    ]
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1] + 1e-9, (
            f"Not monotone decreasing at step {cfg.warmup_steps + i}: "
            f"{vals[i-1]} -> {vals[i]}"
        )


# ---------------------------------------------------------------------------
# 17. get_scheduler "linear" and "polynomial" work
# ---------------------------------------------------------------------------

def test_get_scheduler_linear_and_polynomial():
    from torch.optim.lr_scheduler import LambdaLR
    cfg = tiny_config()
    for name in ("linear", "polynomial"):
        _, optimizer = tiny_model_and_optimizer()
        sched = get_scheduler(name, optimizer, cfg)
        assert isinstance(sched, LambdaLR), f"Expected LambdaLR for scheduler '{name}'"
        # Step through a few iterations without errors
        for _ in range(15):
            sched.step()


# ---------------------------------------------------------------------------
# 18. get_scheduler raises ValueError on unknown name
# ---------------------------------------------------------------------------

def test_get_scheduler_unknown_raises():
    cfg = tiny_config()
    _, optimizer = tiny_model_and_optimizer()
    with pytest.raises(ValueError, match="Unknown scheduler name"):
        get_scheduler("foobar", optimizer, cfg)


# ---------------------------------------------------------------------------
# 19. polynomial_schedule power=2 gives quadratic decay (< linear at midpoint)
# ---------------------------------------------------------------------------

def test_polynomial_power2_less_than_linear():
    cfg = tiny_config()
    step = 55  # past warmup, in decay phase
    linear_val = polynomial_schedule(step, cfg, power=1.0)
    quad_val = polynomial_schedule(step, cfg, power=2.0)
    assert quad_val < linear_val, (
        f"power=2 should decay faster than power=1: {quad_val} vs {linear_val}"
    )


# ---------------------------------------------------------------------------
# 20. linear_schedule at total_steps = min_lr_ratio
# ---------------------------------------------------------------------------

def test_linear_at_total_steps_equals_min_lr():
    cfg = tiny_config()
    val = linear_schedule_with_warmup(cfg.total_steps, cfg)
    assert val == pytest.approx(cfg.min_lr_ratio, abs=1e-6)
