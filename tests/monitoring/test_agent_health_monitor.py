"""Tests for AgentHealthMonitor."""
from __future__ import annotations

import threading
import time

import pytest

from src.monitoring.agent_health_monitor import (
    AgentHealthMonitor,
    AGENT_HEALTH_MONITOR_REGISTRY,
    DEFAULT_AGENT_HEALTH_MONITOR,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegisterAgent:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_register_adds_agent(self):
        self.monitor.register_agent("agent-1")
        health = self.monitor.check_health()
        assert "agent-1" in health

    def test_register_defaults_to_unknown(self):
        self.monitor.register_agent("agent-1")
        health = self.monitor.check_health()
        assert health["agent-1"] == "unknown"

    def test_register_with_custom_timeout(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=0.01)
        self.monitor.record_heartbeat("agent-1")
        time.sleep(0.02)
        health = self.monitor.check_health()
        assert health["agent-1"] == "unhealthy"

    def test_register_overwrites_existing(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=10.0)
        self.monitor.record_heartbeat("agent-1")
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=20.0)
        health = self.monitor.check_health()
        assert health["agent-1"] == "unknown"


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestRecordHeartbeat:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_heartbeat_makes_healthy(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=1.0)
        self.monitor.record_heartbeat("agent-1")
        health = self.monitor.check_health()
        assert health["agent-1"] == "healthy"

    def test_heartbeat_updates_last_seen(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=0.05)
        self.monitor.record_heartbeat("agent-1")
        assert self.monitor.check_health()["agent-1"] == "healthy"
        time.sleep(0.06)
        assert self.monitor.check_health()["agent-1"] == "unhealthy"
        self.monitor.record_heartbeat("agent-1")
        assert self.monitor.check_health()["agent-1"] == "healthy"

    def test_heartbeat_unregistered_raises(self):
        with pytest.raises(KeyError):
            self.monitor.record_heartbeat("missing")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestCheckHealth:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_empty_returns_empty(self):
        assert self.monitor.check_health() == {}

    def test_mixed_statuses(self):
        self.monitor.register_agent("a", heartbeat_timeout_sec=1.0)
        self.monitor.register_agent("b", heartbeat_timeout_sec=1.0)
        self.monitor.record_heartbeat("a")
        health = self.monitor.check_health()
        assert health["a"] == "healthy"
        assert health["b"] == "unknown"


# ---------------------------------------------------------------------------
# Timeout detection
# ---------------------------------------------------------------------------

class TestTimeoutDetection:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_becomes_unhealthy_after_timeout(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=0.01)
        self.monitor.record_heartbeat("agent-1")
        assert self.monitor.check_health()["agent-1"] == "healthy"
        time.sleep(0.02)
        assert self.monitor.check_health()["agent-1"] == "unhealthy"

    def test_stays_healthy_within_timeout(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=1.0)
        self.monitor.record_heartbeat("agent-1")
        assert self.monitor.check_health()["agent-1"] == "healthy"


# ---------------------------------------------------------------------------
# Get unhealthy
# ---------------------------------------------------------------------------

class TestGetUnhealthy:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_no_unhealthy_when_empty(self):
        assert self.monitor.get_unhealthy() == []

    def test_returns_unhealthy_agents(self):
        self.monitor.register_agent("a", heartbeat_timeout_sec=0.01)
        self.monitor.register_agent("b", heartbeat_timeout_sec=1.0)
        self.monitor.record_heartbeat("a")
        self.monitor.record_heartbeat("b")
        time.sleep(0.02)
        unhealthy = self.monitor.get_unhealthy()
        assert unhealthy == ["a"]

    def test_does_not_return_unknown(self):
        self.monitor.register_agent("a", heartbeat_timeout_sec=1.0)
        assert self.monitor.get_unhealthy() == []


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestRemoveAgent:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_remove_deletes_agent(self):
        self.monitor.register_agent("agent-1")
        self.monitor.remove_agent("agent-1")
        assert self.monitor.check_health() == {}

    def test_remove_unregistered_raises(self):
        with pytest.raises(KeyError):
            self.monitor.remove_agent("missing")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def setup_method(self):
        self.monitor = AgentHealthMonitor()

    def test_concurrent_heartbeats(self):
        self.monitor.register_agent("agent-1", heartbeat_timeout_sec=10.0)
        errors: list[Exception] = []

        def heartbeat():
            try:
                for _ in range(100):
                    self.monitor.record_heartbeat("agent-1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=heartbeat) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert self.monitor.check_health()["agent-1"] == "healthy"

    def test_concurrent_register_and_check(self):
        errors: list[Exception] = []

        def worker(agent_id: str):
            try:
                self.monitor.register_agent(agent_id, heartbeat_timeout_sec=10.0)
                self.monitor.record_heartbeat(agent_id)
                _ = self.monitor.check_health()
                _ = self.monitor.get_unhealthy()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"agent-{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(self.monitor.check_health()) == 20


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in AGENT_HEALTH_MONITOR_REGISTRY
        assert isinstance(AGENT_HEALTH_MONITOR_REGISTRY["default"], AgentHealthMonitor)

    def test_default_is_agent_health_monitor(self):
        assert isinstance(DEFAULT_AGENT_HEALTH_MONITOR, AgentHealthMonitor)
