"""Tests for src/training/warmup_scheduler.py."""

import pytest

from src.training.warmup_scheduler import SchedulerType, WarmupScheduler

# ---------------------------------------------------------------------------
# SchedulerType enum
# ---------------------------------------------------------------------------


class TestSchedulerType:
    def test_linear_warmup_value(self):
        assert SchedulerType.LINEAR_WARMUP == "linear_warmup"

    def test_cosine_value(self):
        assert SchedulerType.COSINE == "cosine"

    def test_cosine_with_restarts_value(self):
        assert SchedulerType.COSINE_WITH_RESTARTS == "cosine_with_restarts"

    def test_polynomial_value(self):
        assert SchedulerType.POLYNOMIAL == "polynomial"

    def test_four_members(self):
        assert len(SchedulerType) == 4

    def test_is_str(self):
        assert isinstance(SchedulerType.COSINE, str)


# ---------------------------------------------------------------------------
# WarmupScheduler — basic construction
# ---------------------------------------------------------------------------


class TestWarmupSchedulerConstruction:
    def test_base_lr_stored(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.base_lr == 1e-3

    def test_default_warmup_steps(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.warmup_steps == 100

    def test_default_total_steps(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.total_steps == 10000

    def test_default_scheduler_type(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.scheduler_type == SchedulerType.COSINE

    def test_default_min_lr(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.min_lr == 0.0

    def test_custom_params(self):
        s = WarmupScheduler(
            base_lr=1e-4,
            warmup_steps=50,
            total_steps=5000,
            scheduler_type=SchedulerType.POLYNOMIAL,
            min_lr=1e-6,
        )
        assert s.warmup_steps == 50
        assert s.total_steps == 5000
        assert s.min_lr == 1e-6


# ---------------------------------------------------------------------------
# get_lr — warmup phase
# ---------------------------------------------------------------------------


class TestGetLrWarmup:
    def test_lr_at_step_0_is_zero(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.get_lr(0) == pytest.approx(0.0, abs=1e-10)

    def test_lr_increases_during_warmup(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        lrs = [s.get_lr(i) for i in range(101)]
        for i in range(1, 101):
            assert lrs[i] >= lrs[i - 1]

    def test_lr_at_warmup_steps_approx_base_lr(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100, total_steps=10000)
        # At step == warmup_steps we enter decay; just before it should be close
        lr = s.get_lr(100)
        # At step=100, progress = 0 so cosine gives base_lr
        assert lr == pytest.approx(1e-3, rel=1e-5)

    def test_lr_is_linear_during_warmup(self):
        s = WarmupScheduler(base_lr=1.0, warmup_steps=10)
        assert s.get_lr(5) == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# get_lr — cosine decay
# ---------------------------------------------------------------------------


class TestGetLrCosine:
    def test_cosine_decreasing_after_warmup(self):
        s = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=10,
            total_steps=200,
            scheduler_type=SchedulerType.COSINE,
        )
        lrs = [s.get_lr(i) for i in range(10, 200)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12  # non-increasing

    def test_cosine_approaches_min_lr_at_total_steps(self):
        s = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=1000,
            scheduler_type=SchedulerType.COSINE,
            min_lr=0.0,
        )
        lr_end = s.get_lr(1000)
        assert lr_end == pytest.approx(0.0, abs=1e-9)

    def test_cosine_at_half_decay_approx_half_base(self):
        s = WarmupScheduler(
            base_lr=1.0,
            warmup_steps=0,
            total_steps=1000,
            scheduler_type=SchedulerType.COSINE,
            min_lr=0.0,
        )
        lr_mid = s.get_lr(500)
        # cos(pi * 0.5) = 0 → 0.5 * (1+0) = 0.5
        assert lr_mid == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# get_lr — polynomial decay
# ---------------------------------------------------------------------------


class TestGetLrPolynomial:
    def test_polynomial_decreasing_after_warmup(self):
        s = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=10,
            total_steps=200,
            scheduler_type=SchedulerType.POLYNOMIAL,
        )
        lrs = [s.get_lr(i) for i in range(10, 200)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12

    def test_polynomial_end_value(self):
        s = WarmupScheduler(
            base_lr=1.0,
            warmup_steps=0,
            total_steps=1000,
            scheduler_type=SchedulerType.POLYNOMIAL,
            min_lr=0.0,
        )
        assert s.get_lr(1000) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# get_lr — cosine with restarts
# ---------------------------------------------------------------------------


class TestGetLrCosineRestarts:
    def test_cosine_restarts_resets_within_cycles(self):
        s = WarmupScheduler(
            base_lr=1.0,
            warmup_steps=0,
            total_steps=1000,
            scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
            n_cycles=2,
        )
        # At start of second cycle (~step 500) LR should be near base_lr again
        lr_cycle2_start = s.get_lr(500)
        assert lr_cycle2_start == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# get_lr — linear warmup post-warmup (linear decay)
# ---------------------------------------------------------------------------


class TestGetLrLinearWarmupDecay:
    def test_linear_decay_after_warmup(self):
        s = WarmupScheduler(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=110,
            scheduler_type=SchedulerType.LINEAR_WARMUP,
            min_lr=0.0,
        )
        lrs = [s.get_lr(i) for i in range(10, 111)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestGetLrClamping:
    def test_lr_never_below_min_lr(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=0, total_steps=1000, min_lr=1e-5)
        for step in range(0, 1001, 50):
            assert s.get_lr(step) >= 1e-5

    def test_lr_never_above_base_lr(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100, total_steps=10000)
        for step in range(0, 10001, 500):
            assert s.get_lr(step) <= 1e-3 + 1e-12

    def test_lr_in_range_for_all_scheduler_types(self):
        for stype in SchedulerType:
            s = WarmupScheduler(
                base_lr=1.0,
                warmup_steps=10,
                total_steps=100,
                scheduler_type=stype,
                min_lr=0.0,
            )
            for step in range(0, 101, 5):
                lr = s.get_lr(step)
                assert 0.0 - 1e-9 <= lr <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# warmup_progress
# ---------------------------------------------------------------------------


class TestWarmupProgress:
    def test_warmup_progress_zero_at_step_0(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.warmup_progress(0) == pytest.approx(0.0)

    def test_warmup_progress_one_at_warmup_steps(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.warmup_progress(100) == pytest.approx(1.0)

    def test_warmup_progress_capped_at_one(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.warmup_progress(200) == pytest.approx(1.0)

    def test_warmup_progress_midpoint(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.warmup_progress(50) == pytest.approx(0.5)

    def test_warmup_progress_non_negative(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=100)
        assert s.warmup_progress(0) >= 0.0


# ---------------------------------------------------------------------------
# lr_schedule
# ---------------------------------------------------------------------------


class TestLrSchedule:
    def test_lr_schedule_returns_list(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert isinstance(s.lr_schedule(10), list)

    def test_lr_schedule_correct_length(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert len(s.lr_schedule(50)) == 50

    def test_lr_schedule_zero_steps(self):
        s = WarmupScheduler(base_lr=1e-3)
        assert s.lr_schedule(0) == []

    def test_lr_schedule_values_match_get_lr(self):
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=5, total_steps=20)
        schedule = s.lr_schedule(20)
        for i, lr in enumerate(schedule):
            assert lr == pytest.approx(s.get_lr(i))

    def test_lr_schedule_contains_floats(self):
        s = WarmupScheduler(base_lr=1e-3)
        schedule = s.lr_schedule(5)
        for lr in schedule:
            assert isinstance(lr, float)
