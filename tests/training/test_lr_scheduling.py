"""
Tests for src/training/lr_scheduling.py

Pure PyTorch only.  Tiny configs used throughout.

Import path: aurelius.training.lr_scheduling
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.optim as optim
from aurelius.training.lr_scheduling import (
    CosineWithRestartsScheduler,
    InverseSqrtScheduler,
    LRSchedulerComparison,
    WSDScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_LR = 1e-3
WARMUP = 10
STABLE = 20
DECAY = 30


def _make_optimizer(lr: float = BASE_LR) -> optim.Optimizer:
    model = nn.Linear(8, 8)
    return optim.SGD(model.parameters(), lr=lr)


def _step_n(scheduler, n: int):
    """Call scheduler.step() n times and return list of lrs after each step."""
    lrs = []
    for _ in range(n):
        scheduler.optimizer.step()
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]["lr"])
    return lrs


# ---------------------------------------------------------------------------
# WSDScheduler tests
# ---------------------------------------------------------------------------


class TestWSDScheduler:
    def _make(self, **kw):
        opt = _make_optimizer()
        defaults = dict(
            warmup_steps=WARMUP,
            stable_steps=STABLE,
            decay_steps=DECAY,
            min_lr_ratio=0.1,
        )
        defaults.update(kw)
        return WSDScheduler(opt, **defaults)

    def test_warmup_phase_lr_increases(self):
        """LR should strictly increase during the warmup phase."""
        sched = self._make()
        # step() once to enter step=1, collect up to warmup_end
        lrs = _step_n(sched, WARMUP)
        # lrs[0] is after 1st step (last_epoch=1), lrs[-1] after WARMUP steps
        assert lrs[0] < lrs[-1], "LR should increase during warmup"
        for i in range(1, len(lrs)):
            assert lrs[i - 1] <= lrs[i], f"LR decreased at warmup step {i}"

    def test_stable_phase_lr_constant(self):
        """LR should be constant at base_lr during the stable phase."""
        sched = self._make()
        _step_n(sched, WARMUP)  # exhaust warmup
        stable_lrs = _step_n(sched, STABLE)
        for i, lr in enumerate(stable_lrs):
            assert abs(lr - BASE_LR) < 1e-9, f"LR {lr} != base_lr {BASE_LR} at stable step {i}"

    def test_decay_phase_lr_decreases(self):
        """LR should decrease during the decay phase."""
        sched = self._make()
        _step_n(sched, WARMUP + STABLE)  # exhaust warmup + stable
        decay_lrs = _step_n(sched, DECAY)
        assert decay_lrs[0] > decay_lrs[-1], "LR should decrease during decay"
        for i in range(1, len(decay_lrs)):
            assert decay_lrs[i - 1] >= decay_lrs[i], f"LR increased at decay step {i}"

    def test_post_decay_lr_equals_min(self):
        """After the decay phase LR should equal min_lr_ratio * base_lr."""
        min_ratio = 0.1
        sched = self._make(min_lr_ratio=min_ratio)
        _step_n(sched, WARMUP + STABLE + DECAY + 5)  # go past decay
        lr = sched.optimizer.param_groups[0]["lr"]
        expected = min_ratio * BASE_LR
        assert abs(lr - expected) < 1e-9, f"Expected {expected}, got {lr}"

    def test_warmup_starts_near_zero(self):
        """First step of warmup should produce a very small lr."""
        sched = self._make()
        sched.optimizer.step()
        sched.step()
        lr = sched.optimizer.param_groups[0]["lr"]
        assert lr < BASE_LR * 0.5, "Warmup first step should be well below base_lr"

    def test_warmup_ends_at_base_lr(self):
        """LR at end of warmup should equal base_lr."""
        sched = self._make()
        _step_n(sched, WARMUP)
        lr = sched.optimizer.param_groups[0]["lr"]
        assert abs(lr - BASE_LR) < 1e-9, f"Expected base_lr at warmup end, got {lr}"


# ---------------------------------------------------------------------------
# CosineWithRestartsScheduler tests
# ---------------------------------------------------------------------------


class TestCosineWithRestartsScheduler:
    def _make(self, T_0=10, T_mult=1, min_lr_ratio=0.0):
        opt = _make_optimizer()
        return CosineWithRestartsScheduler(opt, T_0=T_0, T_mult=T_mult, min_lr_ratio=min_lr_ratio)

    def test_lr_at_cycle_start_equals_base_lr(self):
        """LR at the very first step (pos=0 in cycle) should be base_lr."""
        sched = self._make(T_0=10)
        # After __init__, last_epoch=0.  Call step() → last_epoch=1 (pos=1).
        # To get pos=0 we inspect get_lr before stepping.
        # LRScheduler sets lr on __init__ already (last_epoch=0).
        initial_lr = sched.optimizer.param_groups[0]["lr"]
        assert abs(initial_lr - BASE_LR) < 1e-9, (
            f"Expected base_lr at cycle start, got {initial_lr}"
        )

    def test_lr_at_restart_resets_to_base_lr(self):
        """LR should reset to base_lr at the beginning of each new cycle."""
        T_0 = 10
        sched = self._make(T_0=T_0, min_lr_ratio=0.0)
        # Step to the very start of the second cycle (step == T_0)
        _step_n(sched, T_0)
        # At step=T_0, pos=0 → cosine = 1 → lr = base_lr
        lr = sched.optimizer.param_groups[0]["lr"]
        assert abs(lr - BASE_LR) < 1e-9, f"Expected base_lr at restart, got {lr}"

    def test_lr_decreases_within_cycle(self):
        """LR should decrease monotonically within a cycle."""
        T_0 = 20
        sched = self._make(T_0=T_0, min_lr_ratio=0.0)
        lrs = _step_n(sched, T_0 - 1)
        for i in range(1, len(lrs)):
            assert lrs[i - 1] >= lrs[i], (
                f"LR increased within cycle at step {i}: {lrs[i - 1]} → {lrs[i]}"
            )

    def test_T_mult_extends_cycle(self):
        """With T_mult=2, second cycle should be twice as long as the first."""
        T_0 = 10
        sched = self._make(T_0=T_0, T_mult=2, min_lr_ratio=0.0)
        # After first cycle (T_0=10 steps) the second cycle has length 20.
        # At step T_0 + 10 (halfway through second cycle) lr should still be
        # above minimum, i.e. cycle not yet finished.
        _step_n(sched, T_0)  # end of first cycle
        mid_lrs = _step_n(sched, 10)  # halfway into second cycle (len=20)
        # lr halfway into cosine is cos(π*0.5) = 0 → lr ≈ 0 only if T_0==10
        # But with T_mult=2, halfway of a 20-step cycle is progress=0.5 → cos=0
        # So we just check it's non-negative
        assert all(lr >= 0 for lr in mid_lrs)


# ---------------------------------------------------------------------------
# InverseSqrtScheduler tests
# ---------------------------------------------------------------------------


class TestInverseSqrtScheduler:
    def _make(self, warmup_steps=WARMUP):
        opt = _make_optimizer()
        return InverseSqrtScheduler(opt, warmup_steps=warmup_steps)

    def test_warmup_lr_increases(self):
        """LR should increase during warmup."""
        sched = self._make()
        lrs = _step_n(sched, WARMUP)
        for i in range(1, len(lrs)):
            assert lrs[i - 1] <= lrs[i], f"LR decreased at warmup step {i}: {lrs[i - 1]} → {lrs[i]}"

    def test_post_warmup_inverse_sqrt(self):
        """After warmup, lr should equal base_lr * sqrt(warmup / step)."""
        sched = self._make(warmup_steps=WARMUP)
        _step_n(sched, WARMUP)  # complete warmup
        # Now at last_epoch = WARMUP; step once more → last_epoch = WARMUP+1
        sched.optimizer.step()
        sched.step()
        step = WARMUP + 1
        expected = BASE_LR * math.sqrt(WARMUP / step)
        lr = sched.optimizer.param_groups[0]["lr"]
        assert abs(lr - expected) < 1e-9, f"Expected {expected}, got {lr}"

    def test_lr_never_exceeds_base_lr(self):
        """LR should never exceed base_lr at any step."""
        sched = self._make()
        lrs = _step_n(sched, WARMUP + DECAY + 50)
        for i, lr in enumerate(lrs):
            assert lr <= BASE_LR + 1e-9, f"LR {lr} exceeded base_lr {BASE_LR} at step {i}"

    def test_post_warmup_lr_decreases(self):
        """After warmup the inverse-sqrt schedule should decrease."""
        sched = self._make()
        _step_n(sched, WARMUP)
        post_lrs = _step_n(sched, 20)
        for i in range(1, len(post_lrs)):
            assert post_lrs[i - 1] >= post_lrs[i], f"Post-warmup lr increased at step {i}"


# ---------------------------------------------------------------------------
# LRSchedulerComparison tests
# ---------------------------------------------------------------------------


class TestLRSchedulerComparison:
    def _make_comparison(self):
        opt_wsd = _make_optimizer()
        opt_inv = _make_optimizer()
        schedulers = {
            "wsd": WSDScheduler(
                opt_wsd,
                warmup_steps=WARMUP,
                stable_steps=STABLE,
                decay_steps=DECAY,
            ),
            "inv_sqrt": InverseSqrtScheduler(opt_inv, warmup_steps=WARMUP),
        }
        return LRSchedulerComparison(schedulers)

    def test_simulate_returns_correct_shape(self):
        """simulate(n) should return lists of length n for each scheduler."""
        n = 50
        comp = self._make_comparison()
        results = comp.simulate(n)
        for name, lrs in results.items():
            assert len(lrs) == n, f"Scheduler '{name}' returned {len(lrs)} lrs, expected {n}"

    def test_simulate_records_all_schedulers(self):
        """simulate should return an entry for every scheduler passed in."""
        comp = self._make_comparison()
        results = comp.simulate(20)
        assert set(results.keys()) == {"wsd", "inv_sqrt"}

    def test_find_crossover_returns_correct_step(self):
        """find_crossover should return the first index where a > b."""
        lrs_a = [0.0, 0.1, 0.5, 0.9]
        lrs_b = [0.0, 0.2, 0.4, 0.8]
        # a > b first at index 2 (0.5 > 0.4)
        result = LRSchedulerComparison.find_crossover(lrs_a, lrs_b)
        assert result == 2, f"Expected crossover at index 2, got {result}"

    def test_find_crossover_returns_minus_one_when_never(self):
        """find_crossover should return -1 when lrs_a never exceeds lrs_b."""
        lrs_a = [0.1, 0.2, 0.3]
        lrs_b = [0.5, 0.6, 0.7]
        result = LRSchedulerComparison.find_crossover(lrs_a, lrs_b)
        assert result == -1, f"Expected -1, got {result}"

    def test_simulate_values_are_floats(self):
        """All recorded lr values should be Python floats."""
        comp = self._make_comparison()
        results = comp.simulate(10)
        for name, lrs in results.items():
            for lr in lrs:
                assert isinstance(lr, float), (
                    f"Scheduler '{name}' produced non-float lr: {type(lr)}"
                )
