"""Tests for src/compression/pruning_scheduler.py (≥28 tests)."""
from __future__ import annotations

import dataclasses
import pytest

from src.compression.pruning_scheduler import (
    PruningConfig,
    PruningScheduler,
    SparsitySchedule,
    PRUNING_SCHEDULER_REGISTRY,
)


# ---------------------------------------------------------------------------
# PruningConfig
# ---------------------------------------------------------------------------

class TestPruningConfig:
    def test_defaults(self):
        cfg = PruningConfig()
        assert cfg.initial_sparsity == 0.0
        assert cfg.target_sparsity == 0.9
        assert cfg.begin_step == 0
        assert cfg.end_step == 1000
        assert cfg.frequency == 100

    def test_custom(self):
        cfg = PruningConfig(initial_sparsity=0.1, target_sparsity=0.8, begin_step=50,
                            end_step=500, frequency=50)
        assert cfg.begin_step == 50
        assert cfg.frequency == 50

    def test_frozen(self):
        cfg = PruningConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.target_sparsity = 0.5  # type: ignore[misc]

    def test_frozen_begin_step(self):
        cfg = PruningConfig(begin_step=10)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.begin_step = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SparsitySchedule enum
# ---------------------------------------------------------------------------

class TestSparsityScheduleEnum:
    def test_members_exist(self):
        assert SparsitySchedule.CONSTANT
        assert SparsitySchedule.LINEAR
        assert SparsitySchedule.POLYNOMIAL
        assert SparsitySchedule.CUBIC

    def test_values(self):
        assert SparsitySchedule.CONSTANT.value == "constant"
        assert SparsitySchedule.LINEAR.value == "linear"


# ---------------------------------------------------------------------------
# PruningScheduler.sparsity_at — boundary conditions
# ---------------------------------------------------------------------------

class TestSparsityAtBoundaries:
    def _sched(self, schedule: SparsitySchedule = SparsitySchedule.CUBIC) -> PruningScheduler:
        cfg = PruningConfig(initial_sparsity=0.0, target_sparsity=0.9,
                            begin_step=100, end_step=1000)
        return PruningScheduler(config=cfg, schedule=schedule)

    def test_before_begin_returns_initial(self):
        ps = self._sched()
        assert ps.sparsity_at(0) == 0.0
        assert ps.sparsity_at(99) == 0.0

    def test_at_begin_step_returns_initial(self):
        ps = self._sched()
        assert ps.sparsity_at(100) == pytest.approx(0.0, abs=1e-9)

    def test_after_end_returns_target(self):
        ps = self._sched()
        assert ps.sparsity_at(1000) == pytest.approx(0.9)
        assert ps.sparsity_at(9999) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# CONSTANT schedule
# ---------------------------------------------------------------------------

class TestConstantSchedule:
    def _ps(self) -> PruningScheduler:
        cfg = PruningConfig(initial_sparsity=0.3, target_sparsity=0.9,
                            begin_step=0, end_step=1000)
        return PruningScheduler(config=cfg, schedule=SparsitySchedule.CONSTANT)

    def test_always_initial_before(self):
        ps = self._ps()
        assert ps.sparsity_at(0) == pytest.approx(0.3)

    def test_always_initial_during(self):
        ps = self._ps()
        assert ps.sparsity_at(500) == pytest.approx(0.3)

    def test_always_initial_after(self):
        ps = self._ps()
        assert ps.sparsity_at(2000) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# LINEAR schedule
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    def _ps(self) -> PruningScheduler:
        cfg = PruningConfig(initial_sparsity=0.0, target_sparsity=0.9,
                            begin_step=0, end_step=1000)
        return PruningScheduler(config=cfg, schedule=SparsitySchedule.LINEAR)

    def test_midpoint(self):
        ps = self._ps()
        mid = ps.sparsity_at(500)
        assert mid == pytest.approx(0.45, abs=1e-6)

    def test_quarter(self):
        ps = self._ps()
        assert ps.sparsity_at(250) == pytest.approx(0.225, abs=1e-6)

    def test_at_end(self):
        ps = self._ps()
        assert ps.sparsity_at(1000) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# CUBIC schedule
# ---------------------------------------------------------------------------

class TestCubicSchedule:
    def _ps(
        self,
        initial: float = 0.0,
        target: float = 0.9,
        begin: int = 0,
        end: int = 1000,
    ) -> PruningScheduler:
        cfg = PruningConfig(initial_sparsity=initial, target_sparsity=target,
                            begin_step=begin, end_step=end)
        return PruningScheduler(config=cfg, schedule=SparsitySchedule.CUBIC)

    def test_at_end_equals_target(self):
        ps = self._ps()
        assert ps.sparsity_at(1000) == pytest.approx(0.9)

    def test_cubic_formula_midpoint(self):
        ps = self._ps()
        pct = 0.5
        expected = 0.9 - (0.9 - 0.0) * (1.0 - pct) ** 3
        assert ps.sparsity_at(500) == pytest.approx(expected, abs=1e-9)

    def test_cubic_faster_than_linear_early(self):
        ps_cubic = self._ps()
        cfg_lin = PruningConfig(initial_sparsity=0.0, target_sparsity=0.9,
                                begin_step=0, end_step=1000)
        ps_lin = PruningScheduler(config=cfg_lin, schedule=SparsitySchedule.LINEAR)
        # Cubic ramp s = final - (final-initial)*(1-pct)^3 is concave-down initially,
        # so it rises faster than linear at small pct values.
        assert ps_cubic.sparsity_at(200) > ps_lin.sparsity_at(200)

    def test_cubic_at_begin_step_nonzero_begin(self):
        ps = self._ps(begin=100, end=1100)
        assert ps.sparsity_at(99) == pytest.approx(0.0)
        assert ps.sparsity_at(1100) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# POLYNOMIAL schedule
# ---------------------------------------------------------------------------

class TestPolynomialSchedule:
    def test_same_as_cubic(self):
        cfg = PruningConfig(initial_sparsity=0.0, target_sparsity=0.9,
                            begin_step=0, end_step=1000)
        ps_poly = PruningScheduler(config=cfg, schedule=SparsitySchedule.POLYNOMIAL)
        ps_cubic = PruningScheduler(config=cfg, schedule=SparsitySchedule.CUBIC)
        for step in [100, 300, 500, 700, 900]:
            assert ps_poly.sparsity_at(step) == pytest.approx(ps_cubic.sparsity_at(step), abs=1e-12)


# ---------------------------------------------------------------------------
# PruningScheduler.should_prune
# ---------------------------------------------------------------------------

class TestShouldPrune:
    def _ps(self) -> PruningScheduler:
        cfg = PruningConfig(begin_step=100, end_step=1000, frequency=100)
        return PruningScheduler(config=cfg)

    def test_at_begin_step_true(self):
        ps = self._ps()
        assert ps.should_prune(100) is True

    def test_before_begin_false(self):
        ps = self._ps()
        assert ps.should_prune(99) is False

    def test_at_frequency_multiple_true(self):
        ps = self._ps()
        assert ps.should_prune(200) is True
        assert ps.should_prune(500) is True

    def test_non_frequency_step_false(self):
        ps = self._ps()
        assert ps.should_prune(150) is False
        assert ps.should_prune(101) is False

    def test_beyond_end_step_still_true_if_frequency_aligned(self):
        ps = self._ps()
        # step=1100 → 1100 >= 100 and (1100-100) % 100 == 0
        assert ps.should_prune(1100) is True


# ---------------------------------------------------------------------------
# PruningScheduler.schedule_steps
# ---------------------------------------------------------------------------

class TestScheduleSteps:
    def _ps(self) -> PruningScheduler:
        cfg = PruningConfig(begin_step=0, end_step=1000, frequency=100)
        return PruningScheduler(config=cfg)

    def test_returns_list_of_tuples(self):
        ps = self._ps()
        steps = ps.schedule_steps(1000)
        assert isinstance(steps, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in steps)

    def test_first_item_begin_step(self):
        ps = self._ps()
        steps = ps.schedule_steps(1000)
        assert steps[0][0] == 0

    def test_all_steps_are_prune_steps(self):
        ps = self._ps()
        steps = ps.schedule_steps(1000)
        for step, _ in steps:
            assert ps.should_prune(step)

    def test_length(self):
        ps = self._ps()
        steps = ps.schedule_steps(1000)
        # begin=0..end=1000, frequency=100 → 0,100,200,...,1000 = 11 steps
        assert len(steps) == 11

    def test_nonzero_begin(self):
        cfg = PruningConfig(begin_step=200, end_step=800, frequency=100)
        ps = PruningScheduler(config=cfg)
        steps = ps.schedule_steps(800)
        assert steps[0][0] == 200

    def test_empty_if_total_below_begin(self):
        cfg = PruningConfig(begin_step=500, end_step=1000, frequency=100)
        ps = PruningScheduler(config=cfg)
        steps = ps.schedule_steps(400)
        assert steps == []


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in PRUNING_SCHEDULER_REGISTRY

    def test_registry_default_is_class(self):
        assert PRUNING_SCHEDULER_REGISTRY["default"] is PruningScheduler

    def test_registry_instantiable(self):
        cls = PRUNING_SCHEDULER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, PruningScheduler)
