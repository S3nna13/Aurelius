"""Tests for src/protocol/backpressure_controller.py"""

import pytest
from src.protocol.backpressure_controller import (
    BackpressureController,
    BackpressureSignal,
    BackpressureState,
    PROTOCOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# evaluate — signal levels
# ---------------------------------------------------------------------------


def test_evaluate_none_below_soft():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=10, throughput_rps=100.0)
    assert state.signal == BackpressureSignal.NONE
    assert state.suggested_delay_ms == 0.0


def test_evaluate_soft_at_threshold():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=50, throughput_rps=100.0)
    assert state.signal == BackpressureSignal.SOFT
    assert state.suggested_delay_ms == 10.0


def test_evaluate_soft_above_threshold():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=75, throughput_rps=100.0)
    assert state.signal == BackpressureSignal.SOFT


def test_evaluate_hard_at_threshold():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=100, throughput_rps=50.0)
    assert state.signal == BackpressureSignal.HARD
    assert state.suggested_delay_ms == 100.0


def test_evaluate_critical_at_threshold():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=200, throughput_rps=10.0)
    assert state.signal == BackpressureSignal.CRITICAL
    assert state.suggested_delay_ms == 1000.0


def test_evaluate_critical_above_threshold():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=500, throughput_rps=10.0)
    assert state.signal == BackpressureSignal.CRITICAL


# ---------------------------------------------------------------------------
# evaluate — drain_rate propagated
# ---------------------------------------------------------------------------


def test_evaluate_sets_drain_rate():
    ctrl = BackpressureController()
    state = ctrl.evaluate(queue_depth=5, throughput_rps=42.5)
    assert state.drain_rate == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# get_drain_estimate
# ---------------------------------------------------------------------------


def test_drain_estimate_basic():
    ctrl = BackpressureController()
    secs = ctrl.get_drain_estimate(200, 40.0)
    assert secs == pytest.approx(5.0)


def test_drain_estimate_zero_throughput_returns_inf():
    ctrl = BackpressureController()
    assert ctrl.get_drain_estimate(100, 0.0) == float("inf")


def test_drain_estimate_negative_throughput_returns_inf():
    ctrl = BackpressureController()
    assert ctrl.get_drain_estimate(100, -1.0) == float("inf")


# ---------------------------------------------------------------------------
# custom thresholds
# ---------------------------------------------------------------------------


def test_custom_thresholds():
    ctrl = BackpressureController(thresholds={"SOFT": 10, "HARD": 20, "CRITICAL": 30})
    assert ctrl.evaluate(9, 1.0).signal == BackpressureSignal.NONE
    assert ctrl.evaluate(10, 1.0).signal == BackpressureSignal.SOFT
    assert ctrl.evaluate(20, 1.0).signal == BackpressureSignal.HARD
    assert ctrl.evaluate(30, 1.0).signal == BackpressureSignal.CRITICAL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_protocol_registry_contains_backpressure():
    assert "backpressure" in PROTOCOL_REGISTRY
    assert isinstance(PROTOCOL_REGISTRY["backpressure"], BackpressureController)
