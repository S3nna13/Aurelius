"""Integration tests for LossVarianceMonitor in a simulated training loop."""

from __future__ import annotations

import math

from src.training import LossVarianceMonitor


def test_exposed_via_src_training() -> None:
    # Importing via the package namespace must work.
    mon = LossVarianceMonitor(window_size=50)
    assert mon is not None


def test_smooth_loss_50_steps_no_spike() -> None:
    mon = LossVarianceMonitor(window_size=50, spike_threshold=3.0)
    # smooth decaying loss
    for step in range(50):
        loss = 2.0 * math.exp(-0.02 * step) + 0.001 * ((step % 7) - 3)
        mon.record(loss, step=step)

    anomalies = mon.anomalies()
    assert not any(a["type"] == "spike" for a in anomalies)
    stats = mon.stats()
    assert stats.n == 50
    assert stats.std >= 0.0


def test_synthetic_outlier_triggers_spike_detection() -> None:
    mon = LossVarianceMonitor(window_size=50, spike_threshold=3.0)

    for step in range(49):
        loss = 1.0 + 0.01 * math.sin(step * 0.3)
        mon.record(loss, step=step)

    # inject a large outlier
    mon.record(50.0, step=49)

    types = {a["type"] for a in mon.anomalies()}
    assert "spike" in types
    # The spike entry should reference step 49.
    spike_entries = [a for a in mon.anomalies() if a["type"] == "spike"]
    assert any(a["step"] == 49 for a in spike_entries)
