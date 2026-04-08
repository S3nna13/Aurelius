import math
import pytest
import torch
import torch.nn as nn
from src.training.ppo_scheduler import (
    AdaptiveKLConfig,
    AdaptiveKLController,
    FixedKLController,
    PPOWarmupScheduler,
)


# ---------------------------------------------------------------------------
# AdaptiveKLConfig tests
# ---------------------------------------------------------------------------

def test_adaptive_kl_config_defaults():
    cfg = AdaptiveKLConfig()
    assert cfg.target_kl == pytest.approx(0.01)
    assert cfg.initial_kl_coef == pytest.approx(0.2)
    assert cfg.kl_coef_min == pytest.approx(0.0)
    assert cfg.kl_coef_max == pytest.approx(1.0)
    assert cfg.adaptation_horizon == 10


def test_adaptive_kl_initial_coef():
    cfg = AdaptiveKLConfig(initial_kl_coef=0.3)
    ctrl = AdaptiveKLController(cfg)
    assert ctrl.kl_coef == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# AdaptiveKLController behaviour tests
# ---------------------------------------------------------------------------

def test_adaptive_kl_increases_on_high_kl():
    """kl > 2 * target_kl should multiply kl_coef by 1.5."""
    cfg = AdaptiveKLConfig(target_kl=0.01, initial_kl_coef=0.2, adaptation_horizon=1)
    ctrl = AdaptiveKLController(cfg)
    # Feed a KL value well above 2 * target (2 * 0.01 = 0.02)
    result = ctrl.update(0.05)
    expected = min(0.2 * 1.5, cfg.kl_coef_max)
    assert result == pytest.approx(expected)


def test_adaptive_kl_decreases_on_low_kl():
    """kl < 0.5 * target_kl should multiply kl_coef by 0.5."""
    cfg = AdaptiveKLConfig(target_kl=0.01, initial_kl_coef=0.2, adaptation_horizon=1)
    ctrl = AdaptiveKLController(cfg)
    # Feed a KL value well below 0.5 * target (0.5 * 0.01 = 0.005)
    result = ctrl.update(0.001)
    expected = max(0.2 * 0.5, cfg.kl_coef_min)
    assert result == pytest.approx(expected)


def test_adaptive_kl_no_change_in_zone():
    """0.5*target <= kl <= 2*target should leave kl_coef unchanged."""
    cfg = AdaptiveKLConfig(target_kl=0.01, initial_kl_coef=0.2, adaptation_horizon=1)
    ctrl = AdaptiveKLController(cfg)
    # KL exactly at target — in the dead zone
    result = ctrl.update(0.01)
    assert result == pytest.approx(0.2)


def test_adaptive_kl_respects_min():
    """kl_coef should never go below kl_coef_min."""
    cfg = AdaptiveKLConfig(
        target_kl=0.01,
        initial_kl_coef=0.0001,
        kl_coef_min=0.05,
        adaptation_horizon=1,
    )
    ctrl = AdaptiveKLController(cfg)
    # Very low KL → would halve, but floor is kl_coef_min
    result = ctrl.update(0.0)
    assert result >= cfg.kl_coef_min


def test_adaptive_kl_respects_max():
    """kl_coef should never exceed kl_coef_max."""
    cfg = AdaptiveKLConfig(
        target_kl=0.01,
        initial_kl_coef=0.9,
        kl_coef_max=1.0,
        adaptation_horizon=1,
    )
    ctrl = AdaptiveKLController(cfg)
    # Very high KL → would multiply by 1.5, but ceiling is kl_coef_max
    result = ctrl.update(1.0)
    assert result <= cfg.kl_coef_max


def test_adaptive_kl_updates_on_horizon():
    """kl_coef should only change every adaptation_horizon steps."""
    horizon = 5
    cfg = AdaptiveKLConfig(
        target_kl=0.01,
        initial_kl_coef=0.2,
        adaptation_horizon=horizon,
    )
    ctrl = AdaptiveKLController(cfg)
    initial_coef = ctrl.kl_coef

    # Feed high-KL values but stop one short of the horizon
    for i in range(horizon - 1):
        result = ctrl.update(0.5)  # high KL > 2 * target
        assert result == pytest.approx(initial_coef), (
            f"kl_coef should not change before horizon; step {i+1}"
        )

    # The horizon-th update should trigger adaptation
    result = ctrl.update(0.5)
    assert result != pytest.approx(initial_coef), "kl_coef should change at horizon"
    assert result == pytest.approx(min(initial_coef * 1.5, cfg.kl_coef_max))


def test_adaptive_kl_get_stats():
    """get_stats() must return the required keys."""
    cfg = AdaptiveKLConfig()
    ctrl = AdaptiveKLController(cfg)
    stats = ctrl.get_stats()
    assert set(stats.keys()) == {"kl_coef", "mean_kl", "n_updates", "step"}
    assert isinstance(stats["kl_coef"], float)
    assert isinstance(stats["mean_kl"], float)
    assert isinstance(stats["n_updates"], int)
    assert isinstance(stats["step"], int)


# ---------------------------------------------------------------------------
# FixedKLController tests
# ---------------------------------------------------------------------------

def test_fixed_kl_unchanged():
    """FixedKLController.update() must always return the original kl_coef."""
    coef = 0.3
    ctrl = FixedKLController(kl_coef=coef)
    for kl in [0.0, 0.001, 0.5, 10.0]:
        assert ctrl.update(kl) == pytest.approx(coef)
    stats = ctrl.get_stats()
    assert stats["kl_coef"] == pytest.approx(coef)


# ---------------------------------------------------------------------------
# PPOWarmupScheduler tests
# ---------------------------------------------------------------------------

def _make_optimizer(lr: float = 1e-3):
    model = nn.Linear(4, 4)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def test_ppo_warmup_scheduler_increases_lr():
    """LR must monotonically increase during warmup phase."""
    base_lr = 1e-3
    n_warmup = 10
    n_total = 100
    opt = _make_optimizer(base_lr)
    sched = PPOWarmupScheduler(opt, n_warmup_steps=n_warmup, n_total_steps=n_total)

    lrs = []
    for _ in range(n_warmup):
        lrs.append(sched.step())

    # All warmup LRs should be monotonically non-decreasing
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1)), (
        f"LR not monotone during warmup: {lrs}"
    )
    # First LR should be less than base_lr
    assert lrs[0] < base_lr
    # Final warmup LR should be close to base_lr
    assert lrs[-1] == pytest.approx(base_lr, rel=0.2)


def test_ppo_warmup_scheduler_cosine_decay():
    """LR must decrease during cosine decay phase after warmup."""
    base_lr = 1e-3
    n_warmup = 5
    n_total = 50
    opt = _make_optimizer(base_lr)
    sched = PPOWarmupScheduler(opt, n_warmup_steps=n_warmup, n_total_steps=n_total)

    # Burn through warmup
    for _ in range(n_warmup):
        sched.step()

    # Collect post-warmup LRs
    post_warmup_lrs = []
    for _ in range(n_total - n_warmup):
        post_warmup_lrs.append(sched.step())

    # The cosine phase should produce a generally decreasing sequence
    assert post_warmup_lrs[0] > post_warmup_lrs[-1], (
        "LR should decrease over cosine decay phase"
    )
    # Final LR should be close to min_lr (default min_lr_ratio=0.1 → 1e-4)
    expected_min = base_lr * 0.1
    assert post_warmup_lrs[-1] == pytest.approx(expected_min, rel=0.05)
