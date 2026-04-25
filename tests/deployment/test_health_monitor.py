"""Tests for deployment health monitor."""
from __future__ import annotations

import pytest

from src.deployment.health_monitor import DeploymentHealthMonitor, HealthMetric


class TestDeploymentHealthMonitor:
    def test_record_and_status(self):
        mon = DeploymentHealthMonitor()
        mon.record(HealthMetric("latency", 100.0, threshold=200.0))
        assert mon.status()["latency"] is True

    def test_failure_detected(self):
        mon = DeploymentHealthMonitor()
        mon.record(HealthMetric("error_rate", 10.0, threshold=5.0))
        assert len(mon.failures()) == 1

    def test_all_healthy(self):
        mon = DeploymentHealthMonitor()
        mon.record(HealthMetric("cpu", 50.0, threshold=90.0))
        mon.record(HealthMetric("mem", 60.0, threshold=80.0))
        assert len(mon.failures()) == 0

    def test_recent(self):
        mon = DeploymentHealthMonitor()
        for i in range(5):
            mon.record(HealthMetric(f"m{i}", float(i), threshold=10.0))
        assert len(mon.recent(3)) == 3

    def test_clear(self):
        mon = DeploymentHealthMonitor()
        mon.record(HealthMetric("x", 1.0, threshold=2.0))
        mon.clear()
        assert len(mon.failures()) == 0