"""Tests for quantization_scheduler."""

from __future__ import annotations

import pytest

from src.compression.quantization_scheduler import (
    QUANTIZATION_SCHEDULER_REGISTRY,
    QuantizationScheduler,
    QuantizationStep,
    QuantSchedule,
)

# --- enum ---


def test_schedule_linear():
    assert QuantSchedule.LINEAR == "linear"


def test_schedule_cosine():
    assert QuantSchedule.COSINE == "cosine"


def test_schedule_step():
    assert QuantSchedule.STEP == "step"


def test_schedule_constant():
    assert QuantSchedule.CONSTANT == "constant"


# --- QuantizationStep ---


def test_step_fields():
    s = QuantizationStep(step=5, bits=8.0, scale=2.0)
    assert s.step == 5
    assert s.bits == 8.0
    assert s.scale == 2.0


def test_step_frozen():
    s = QuantizationStep(step=5, bits=8.0, scale=2.0)
    with pytest.raises(Exception):
        s.bits = 4.0  # type: ignore[misc]


# --- init ---


def test_init_defaults():
    q = QuantizationScheduler()
    assert q.initial_bits == 16.0
    assert q.final_bits == 4.0
    assert q.total_steps == 1000
    assert q.schedule == QuantSchedule.LINEAR


def test_init_custom():
    q = QuantizationScheduler(
        initial_bits=32.0,
        final_bits=8.0,
        total_steps=500,
        schedule=QuantSchedule.COSINE,
    )
    assert q.initial_bits == 32.0
    assert q.schedule == QuantSchedule.COSINE


# --- LINEAR ---


def test_linear_start_equals_initial():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR)
    assert q.bits_at(0) == 16.0


def test_linear_end_equals_final():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR, total_steps=1000)
    assert q.bits_at(1000) == pytest.approx(4.0)


def test_linear_monotonic_decrease():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR, total_steps=1000)
    prev = q.bits_at(0)
    for s in range(100, 1001, 100):
        cur = q.bits_at(s)
        assert cur <= prev + 1e-9
        prev = cur


def test_linear_midpoint():
    q = QuantizationScheduler(
        initial_bits=16.0,
        final_bits=4.0,
        total_steps=1000,
        schedule=QuantSchedule.LINEAR,
    )
    assert q.bits_at(500) == pytest.approx(10.0)


def test_linear_clamped_beyond_total():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR, total_steps=100)
    assert q.bits_at(10_000) == pytest.approx(4.0)


# --- COSINE ---


def test_cosine_start_equals_initial():
    q = QuantizationScheduler(schedule=QuantSchedule.COSINE)
    assert q.bits_at(0) == pytest.approx(16.0)


def test_cosine_end_equals_final():
    q = QuantizationScheduler(schedule=QuantSchedule.COSINE, total_steps=1000)
    assert q.bits_at(1000) == pytest.approx(4.0)


def test_cosine_smooth_decreasing():
    q = QuantizationScheduler(schedule=QuantSchedule.COSINE, total_steps=1000)
    vals = [q.bits_at(s) for s in range(0, 1001, 100)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9


def test_cosine_midpoint_is_average():
    q = QuantizationScheduler(
        initial_bits=16.0,
        final_bits=4.0,
        total_steps=1000,
        schedule=QuantSchedule.COSINE,
    )
    assert q.bits_at(500) == pytest.approx(10.0, abs=1e-6)


# --- STEP ---


def test_step_schedule_initial_plateau():
    q = QuantizationScheduler(
        schedule=QuantSchedule.STEP,
        total_steps=1000,
        step_interval=100,
    )
    assert q.bits_at(50) == 16.0


def test_step_schedule_decreases():
    q = QuantizationScheduler(
        initial_bits=16.0,
        final_bits=4.0,
        total_steps=1000,
        schedule=QuantSchedule.STEP,
        step_interval=100,
    )
    assert q.bits_at(1000) == pytest.approx(4.0)


def test_step_schedule_clamped():
    q = QuantizationScheduler(
        schedule=QuantSchedule.STEP,
        total_steps=1000,
        step_interval=100,
    )
    v = q.bits_at(100_000)
    assert v >= q.final_bits - 1e-9


# --- CONSTANT ---


def test_constant_always_initial():
    q = QuantizationScheduler(schedule=QuantSchedule.CONSTANT)
    for s in (0, 100, 500, 1000, 10_000):
        assert q.bits_at(s) == 16.0


# --- should_quantize ---


def test_should_quantize_false_at_start():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR)
    assert q.should_quantize(0) is False


def test_should_quantize_true_later():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR, total_steps=1000)
    assert q.should_quantize(999) is True


def test_should_quantize_constant_never():
    q = QuantizationScheduler(schedule=QuantSchedule.CONSTANT)
    assert q.should_quantize(500) is False


# --- get_step ---


def test_get_step_returns_type():
    q = QuantizationScheduler()
    s = q.get_step(0)
    assert isinstance(s, QuantizationStep)


def test_get_step_scale_formula():
    q = QuantizationScheduler(
        initial_bits=16.0,
        final_bits=4.0,
        total_steps=1000,
        schedule=QuantSchedule.LINEAR,
    )
    s = q.get_step(500)
    assert s.scale == pytest.approx(16.0 / s.bits)


def test_get_step_records_step():
    q = QuantizationScheduler()
    s = q.get_step(42)
    assert s.step == 42


# --- schedule_summary ---


def test_summary_default_length():
    q = QuantizationScheduler()
    summary = q.schedule_summary()
    assert len(summary) == 10


def test_summary_custom_length():
    q = QuantizationScheduler()
    summary = q.schedule_summary(num_checkpoints=5)
    assert len(summary) == 5


def test_summary_all_quantization_steps():
    q = QuantizationScheduler()
    summary = q.schedule_summary(num_checkpoints=4)
    assert all(isinstance(s, QuantizationStep) for s in summary)


def test_summary_first_is_start():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR)
    summary = q.schedule_summary(num_checkpoints=5)
    assert summary[0].step == 0
    assert summary[0].bits == pytest.approx(16.0)


def test_summary_last_is_end():
    q = QuantizationScheduler(schedule=QuantSchedule.LINEAR, total_steps=1000)
    summary = q.schedule_summary(num_checkpoints=5)
    assert summary[-1].step == 1000


# --- registry ---


def test_registry_has_default():
    assert "default" in QUANTIZATION_SCHEDULER_REGISTRY


def test_registry_constructs():
    cls = QUANTIZATION_SCHEDULER_REGISTRY["default"]
    assert isinstance(cls(), QuantizationScheduler)
