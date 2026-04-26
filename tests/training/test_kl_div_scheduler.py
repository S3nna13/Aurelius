"""Tests for src/training/kl_div_scheduler.py"""

import pytest

from src.training.kl_div_scheduler import (
    AdaptiveKLScheduler,
    CyclicKLScheduler,
    KLSchedulerConfig,
    WarmupKLScheduler,
    create_kl_scheduler,
)

# ---------------------------------------------------------------------------
# Test 1: KLSchedulerConfig defaults
# ---------------------------------------------------------------------------


def test_kl_scheduler_config_defaults():
    cfg = KLSchedulerConfig()
    assert cfg.scheduler_type == "adaptive"
    assert cfg.target_kl == pytest.approx(0.02)
    assert cfg.initial_beta == pytest.approx(0.2)
    assert cfg.warmup_steps == 100
    assert cfg.min_beta == pytest.approx(0.001)
    assert cfg.max_beta == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Test 2: AdaptiveKLScheduler returns initial_beta before any updates
# ---------------------------------------------------------------------------


def test_adaptive_initial_beta():
    sched = AdaptiveKLScheduler(initial_beta=0.3)
    assert sched.get_beta() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 3: Beta increases when KL consistently above target
# ---------------------------------------------------------------------------


def test_adaptive_beta_increases_on_high_kl():
    """Feed KL values well above target * 1.5 for a full horizon; beta must rise."""
    target = 0.02
    initial = 0.2
    horizon = 5
    factor = 2.0
    sched = AdaptiveKLScheduler(
        target_kl=target,
        initial_beta=initial,
        kl_horizon=horizon,
        adjustment_factor=factor,
    )
    # KL = 0.1 >> target * 1.5 = 0.03
    high_kl = target * 3.0
    for _ in range(horizon):
        beta = sched.update(high_kl)

    expected = min(initial * factor, sched.max_beta)
    assert beta == pytest.approx(expected)
    assert beta > initial


# ---------------------------------------------------------------------------
# Test 4: Beta decreases when KL consistently below target
# ---------------------------------------------------------------------------


def test_adaptive_beta_decreases_on_low_kl():
    """Feed KL values well below target / 1.5 for a full horizon; beta must fall."""
    target = 0.02
    initial = 0.2
    horizon = 5
    factor = 2.0
    sched = AdaptiveKLScheduler(
        target_kl=target,
        initial_beta=initial,
        kl_horizon=horizon,
        adjustment_factor=factor,
    )
    # KL = 0.001 << target / 1.5 ≈ 0.013
    low_kl = target / 4.0
    for _ in range(horizon):
        beta = sched.update(low_kl)

    expected = max(initial / factor, sched.min_beta)
    assert beta == pytest.approx(expected)
    assert beta < initial


# ---------------------------------------------------------------------------
# Test 5: Beta clips to [min_beta, max_beta]
# ---------------------------------------------------------------------------


def test_adaptive_beta_clips_to_bounds():
    # Test upper clip
    sched_high = AdaptiveKLScheduler(
        target_kl=0.01,
        initial_beta=9.0,
        kl_horizon=1,
        adjustment_factor=10.0,
        max_beta=10.0,
    )
    beta_high = sched_high.update(1.0)  # very high KL → would exceed max
    assert beta_high <= 10.0

    # Test lower clip
    sched_low = AdaptiveKLScheduler(
        target_kl=0.1,
        initial_beta=0.002,
        kl_horizon=1,
        adjustment_factor=10.0,
        min_beta=0.001,
    )
    beta_low = sched_low.update(0.0)  # very low KL → would go below min
    assert beta_low >= 0.001


# ---------------------------------------------------------------------------
# Test 6: reset() restores initial state
# ---------------------------------------------------------------------------


def test_adaptive_reset_restores_state():
    sched = AdaptiveKLScheduler(
        target_kl=0.02,
        initial_beta=0.2,
        kl_horizon=3,
    )
    # Mutate state
    for _ in range(6):
        sched.update(1.0)  # high KL → beta will increase

    assert sched.get_beta() != pytest.approx(0.2)

    sched.reset()

    assert sched.get_beta() == pytest.approx(0.2)
    stats = sched.get_stats()
    assert stats["n_updates"] == 0
    assert stats["mean_kl"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 7: get_stats returns dict with all required keys
# ---------------------------------------------------------------------------


def test_adaptive_get_stats_keys():
    sched = AdaptiveKLScheduler()
    stats = sched.get_stats()
    required_keys = {"beta", "mean_kl", "target_kl", "n_updates"}
    assert required_keys == set(stats.keys())
    # Sanity-check types
    assert isinstance(stats["beta"], float)
    assert isinstance(stats["mean_kl"], float)
    assert isinstance(stats["target_kl"], float)
    assert isinstance(stats["n_updates"], int)


# ---------------------------------------------------------------------------
# Test 8: CyclicKLScheduler stays within [base_beta, max_beta]
# ---------------------------------------------------------------------------


def test_cyclic_stays_in_range():
    base, maximum = 0.1, 1.0
    for mode in ("triangular", "cosine", "step"):
        sched = CyclicKLScheduler(base_beta=base, max_beta=maximum, cycle_steps=50, mode=mode)
        for _ in range(200):
            beta = sched.step()
            assert base <= beta <= maximum, f"mode={mode}: beta={beta} out of [{base}, {maximum}]"


# ---------------------------------------------------------------------------
# Test 9: CyclicKLScheduler completes a full cycle in cycle_steps
# ---------------------------------------------------------------------------


def test_cyclic_full_cycle():
    """After cycle_steps the triangular scheduler should return the same beta
    as it had at step 0, demonstrating periodicity."""
    sched = CyclicKLScheduler(
        base_beta=0.1,
        max_beta=1.0,
        cycle_steps=100,
        mode="triangular",
    )
    # Collect one full cycle
    betas = [sched.step() for _ in range(100)]
    # Collect another full cycle
    betas2 = [sched.step() for _ in range(100)]

    for i, (b1, b2) in enumerate(zip(betas, betas2)):
        assert b1 == pytest.approx(b2, abs=1e-9), f"Cycle mismatch at position {i}: {b1} vs {b2}"


# ---------------------------------------------------------------------------
# Test 10: WarmupKLScheduler starts at start_beta
# ---------------------------------------------------------------------------


def test_warmup_starts_at_start_beta():
    sched = WarmupKLScheduler(start_beta=0.0, end_beta=0.1, warmup_steps=100)
    first_beta = sched.step()
    assert first_beta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11: WarmupKLScheduler reaches end_beta after warmup_steps
# ---------------------------------------------------------------------------


def test_warmup_reaches_end_beta():
    end = 0.1
    warmup = 50
    sched = WarmupKLScheduler(start_beta=0.0, end_beta=end, warmup_steps=warmup)
    beta = None
    for _ in range(warmup):
        beta = sched.step()
    # The last warmup step lands at end_beta
    assert beta == pytest.approx(end, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 12: get_schedule returns list of correct length
# ---------------------------------------------------------------------------


def test_warmup_get_schedule_length():
    sched = WarmupKLScheduler(start_beta=0.0, end_beta=0.1, warmup_steps=20)
    n = 150
    schedule = sched.get_schedule(n)
    assert isinstance(schedule, list)
    assert len(schedule) == n
    # All values must be floats in a sensible range
    for v in schedule:
        assert isinstance(v, float)
        assert v >= 0.0


# ---------------------------------------------------------------------------
# Test 13: create_kl_scheduler returns correct type based on config
# ---------------------------------------------------------------------------


def test_create_kl_scheduler_types():
    for stype, expected_cls in [
        ("adaptive", AdaptiveKLScheduler),
        ("cyclic", CyclicKLScheduler),
        ("warmup", WarmupKLScheduler),
    ]:
        cfg = KLSchedulerConfig(scheduler_type=stype)
        sched = create_kl_scheduler(cfg)
        assert isinstance(sched, expected_cls), (
            f"Expected {expected_cls.__name__} for scheduler_type='{stype}', "
            f"got {type(sched).__name__}"
        )


def test_create_kl_scheduler_unknown_type_raises():
    cfg = KLSchedulerConfig(scheduler_type="unknown_type")
    with pytest.raises(ValueError, match="Unknown scheduler_type"):
        create_kl_scheduler(cfg)
