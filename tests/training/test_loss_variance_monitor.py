"""Unit tests for src.training.loss_variance_monitor."""

from __future__ import annotations

import math
import statistics
import warnings

import pytest

from src.training.loss_variance_monitor import LossStats, LossVarianceMonitor


def test_stats_on_known_sequence_hand_computed() -> None:
    mon = LossVarianceMonitor(window_size=10)
    seq = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, v in enumerate(seq):
        mon.record(v, step=i)
    s = mon.stats()
    # hand-computed: mean = 3, variance (n-1) = 2.5, std = sqrt(2.5)
    assert s.n == 5
    assert math.isclose(s.mean, 3.0, rel_tol=1e-12)
    assert math.isclose(s.variance, 2.5, rel_tol=1e-12)
    assert math.isclose(s.std, math.sqrt(2.5), rel_tol=1e-12)
    assert s.min == 1.0
    assert s.max == 5.0


def test_mean_equals_sum_over_n() -> None:
    mon = LossVarianceMonitor(window_size=50)
    vals = [0.5, 1.5, 2.25, 3.75, 7.0, 0.1]
    for i, v in enumerate(vals):
        mon.record(v, step=i)
    assert math.isclose(mon.stats().mean, sum(vals) / len(vals), rel_tol=1e-12)


def test_variance_welford_matches_statistics_module() -> None:
    mon = LossVarianceMonitor(window_size=100)
    vals = [1e8 + i for i in range(50)]  # large offset stresses naive formulas
    for i, v in enumerate(vals):
        mon.record(v, step=i)
    ref = statistics.variance(vals)
    assert math.isclose(mon.stats().variance, ref, rel_tol=1e-6)


def test_cv_equals_std_over_mean() -> None:
    mon = LossVarianceMonitor(window_size=50)
    for i, v in enumerate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]):
        mon.record(v, step=i)
    s = mon.stats()
    assert math.isclose(s.coefficient_of_variation, s.std / s.mean, rel_tol=1e-12)


def test_is_spike_true_on_outlier() -> None:
    mon = LossVarianceMonitor(window_size=100, spike_threshold=3.0)
    for i in range(50):
        mon.record(1.0 + 0.01 * ((i % 5) - 2), step=i)
    assert mon.is_spike(100.0) is True


def test_is_spike_false_on_normal() -> None:
    mon = LossVarianceMonitor(window_size=100, spike_threshold=3.0)
    for i in range(50):
        mon.record(1.0 + 0.01 * ((i % 5) - 2), step=i)
    assert mon.is_spike(1.02) is False


def test_is_diverging_true_on_monotone_increasing() -> None:
    mon = LossVarianceMonitor(window_size=200, divergence_threshold=2.0)
    # first 20 small, next 20 huge
    for i in range(20):
        mon.record(0.1, step=i)
    for i in range(20, 40):
        mon.record(100.0 * (i - 19), step=i)
    assert mon.is_diverging(recent_window=20) is True


def test_is_diverging_false_on_decreasing() -> None:
    mon = LossVarianceMonitor(window_size=200, divergence_threshold=2.0)
    for i in range(40):
        mon.record(100.0 - i, step=i)
    assert mon.is_diverging(recent_window=20) is False


def test_reset_clears_state() -> None:
    mon = LossVarianceMonitor(window_size=10)
    for i in range(5):
        mon.record(float(i), step=i)
    # trigger an anomaly too
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mon.record(float("nan"), step=99)

    mon.reset()
    s = mon.stats()
    assert s.n == 0
    assert s.mean == 0.0
    assert mon.anomalies() == []


def test_anomalies_list_populated_on_spike_and_nan() -> None:
    mon = LossVarianceMonitor(window_size=100, spike_threshold=3.0)
    for i in range(50):
        mon.record(1.0 + 0.01 * ((i % 5) - 2), step=i)
    mon.record(1000.0, step=50)  # spike
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mon.record(float("nan"), step=51)
    types = {a["type"] for a in mon.anomalies()}
    assert "spike" in types
    assert "nan" in types


def test_empty_monitor_returns_zero_stats() -> None:
    mon = LossVarianceMonitor()
    s = mon.stats()
    assert s == LossStats(
        mean=0.0,
        variance=0.0,
        std=0.0,
        coefficient_of_variation=0.0,
        min=0.0,
        max=0.0,
        n=0,
    )


def test_nan_loss_handled_with_warning() -> None:
    mon = LossVarianceMonitor()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mon.record(float("nan"), step=0)
    assert any(issubclass(rec.category, RuntimeWarning) for rec in w)
    assert mon.stats().n == 0
    assert mon.anomalies()[-1]["type"] == "nan"


def test_inf_loss_handled_with_warning() -> None:
    mon = LossVarianceMonitor()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mon.record(float("inf"), step=0)
    assert any(issubclass(rec.category, RuntimeWarning) for rec in w)
    assert mon.stats().n == 0
    assert mon.anomalies()[-1]["type"] == "inf"


def test_determinism_same_input_same_output() -> None:
    vals = [0.1 * i for i in range(73)]
    a = LossVarianceMonitor(window_size=50)
    b = LossVarianceMonitor(window_size=50)
    for i, v in enumerate(vals):
        a.record(v, step=i)
        b.record(v, step=i)
    assert a.stats() == b.stats()
    assert a.anomalies() == b.anomalies()


def test_window_size_honored_older_entries_dropped() -> None:
    mon = LossVarianceMonitor(window_size=5)
    for i in range(20):
        mon.record(float(i), step=i)
    s = mon.stats()
    assert s.n == 5
    # window should contain last 5 values: 15..19
    assert s.min == 15.0
    assert s.max == 19.0
    assert math.isclose(s.mean, sum(range(15, 20)) / 5, rel_tol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
