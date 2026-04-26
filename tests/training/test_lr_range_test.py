"""Unit tests for src.training.lr_range_test."""

from __future__ import annotations

import math
import random

import pytest

from src.training.lr_range_test import LRRangeTest, LRRangeTestResult


def _u_shaped_fn(lr: float) -> float:
    # Realistic LR-range-test curve: flat plateau at low LR, a steep
    # descent through the "good" range, then a sharp rise past the
    # optimum. Minimum is around lr ~ 1e-3.
    log_lr = math.log10(lr)
    # Sigmoid-style plateau for the low-LR side, quadratic rise after
    # the optimum at log_lr = -3.
    plateau = 4.0 / (1.0 + math.exp(-(-log_lr - 3.0)))
    rise = max(0.0, (log_lr + 3.0)) ** 2
    return plateau + rise + 0.1


def test_u_shaped_loss_best_lr_in_middle_range() -> None:
    test = LRRangeTest(
        _u_shaped_fn,
        lr_start=1e-7,
        lr_end=1.0,
        num_steps=80,
        smooth_factor=0.5,
        divergence_threshold=0.0,
    )
    result = test.run()
    assert isinstance(result, LRRangeTestResult)
    # Steepest descent of a U-shape happens before the minimum, in the
    # low/mid LR range.
    assert 1e-6 <= result.best_lr <= 1e-2


def test_num_steps_small_runs_fast() -> None:
    result = LRRangeTest(lambda lr: 1.0, num_steps=10).run()
    assert len(result.lrs) == 10
    assert len(result.losses) == 10


def test_lr_start_not_less_than_lr_end_raises() -> None:
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, lr_start=1.0, lr_end=1e-3)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, lr_start=1.0, lr_end=1.0)


def test_invalid_smooth_factor_raises() -> None:
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, smooth_factor=-0.1)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, smooth_factor=1.0)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, smooth_factor=2.0)


def test_divergence_detection_stops_early() -> None:
    calls = {"n": 0}

    def exploding(lr: float) -> float:
        calls["n"] += 1
        if calls["n"] < 20:
            return 1.0
        return 1.0 + (calls["n"] - 19) * 50.0

    test = LRRangeTest(
        exploding,
        lr_start=1e-6,
        lr_end=1.0,
        num_steps=200,
        smooth_factor=0.5,
        divergence_threshold=4.0,
    )
    result = test.run()
    assert result.divergence_lr is not None
    assert len(result.losses) < 200


def test_lrs_geometrically_spaced() -> None:
    test = LRRangeTest(lambda lr: 1.0, lr_start=1e-6, lr_end=1.0, num_steps=50)
    result = test.run()
    ratios = [result.lrs[i + 1] / result.lrs[i] for i in range(len(result.lrs) - 1)]
    # All consecutive ratios should be (approximately) equal for a
    # geometric schedule.
    assert all(math.isclose(r, ratios[0], rel_tol=1e-9) for r in ratios)


def test_losses_length_matches_lrs() -> None:
    result = LRRangeTest(lambda lr: lr, num_steps=33).run()
    assert len(result.losses) == len(result.lrs)


def test_determinism_with_seeded_train_step_fn() -> None:
    def make_fn(seed: int):
        rng = random.Random(seed)

        def fn(lr: float) -> float:
            return rng.random() + lr

        return fn

    r1 = LRRangeTest(make_fn(42), num_steps=25, divergence_threshold=0.0).run()
    r2 = LRRangeTest(make_fn(42), num_steps=25, divergence_threshold=0.0).run()
    assert r1.lrs == r2.lrs
    assert r1.losses == r2.losses
    assert r1.best_lr == r2.best_lr


def test_zero_divergence_threshold_never_triggers_early_stop() -> None:
    def exploding(lr: float) -> float:
        return 1e12 * lr  # blows up hard at high lr

    result = LRRangeTest(
        exploding,
        lr_start=1e-6,
        lr_end=1.0,
        num_steps=50,
        divergence_threshold=0.0,
    ).run()
    assert result.divergence_lr is None
    assert len(result.losses) == 50


def test_best_lr_div_10_is_best_lr_over_ten() -> None:
    result = LRRangeTest(
        _u_shaped_fn,
        lr_start=1e-7,
        lr_end=1.0,
        num_steps=40,
        divergence_threshold=0.0,
    ).run()
    assert math.isclose(result.best_lr_div_10, result.best_lr / 10.0, rel_tol=1e-12)


def test_exception_in_train_step_fn_propagates() -> None:
    def boom(lr: float) -> float:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        LRRangeTest(boom, num_steps=10).run()


def test_monotonically_increasing_loss_populates_divergence_lr() -> None:
    calls = {"n": 0}

    def rising(lr: float) -> float:
        calls["n"] += 1
        return float(calls["n"]) ** 2

    result = LRRangeTest(
        rising,
        lr_start=1e-6,
        lr_end=1.0,
        num_steps=200,
        smooth_factor=0.5,
        divergence_threshold=4.0,
    ).run()
    assert result.divergence_lr is not None


def test_monotonically_decreasing_loss_best_lr_near_end() -> None:
    calls = {"n": 0}

    def falling(lr: float) -> float:
        calls["n"] += 1
        return 100.0 - calls["n"]

    result = LRRangeTest(
        falling,
        lr_start=1e-6,
        lr_end=1.0,
        num_steps=50,
        divergence_threshold=0.0,
    ).run()
    # Steepest descent is at the last step; best_lr should match the
    # final LR.
    assert result.best_lr == result.lrs[-1]


def test_num_steps_zero_or_negative_raises() -> None:
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, num_steps=0)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, num_steps=-5)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, num_steps=1)


def test_lr_start_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, lr_start=0.0, lr_end=1.0)
    with pytest.raises(ValueError):
        LRRangeTest(lambda lr: 1.0, lr_start=-1e-6, lr_end=1.0)
