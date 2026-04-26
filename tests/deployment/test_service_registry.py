"""Tests for service registry."""

from __future__ import annotations

from src.deployment.service_registry import ServiceInstance, ServiceRegistry


class TestServiceRegistry:
    def test_register_and_discover(self):
        sr = ServiceRegistry()
        sr.register(ServiceInstance("api", "localhost", 8080))
        instances = sr.discover("api")
        assert len(instances) == 1

    def test_unregister(self):
        sr = ServiceRegistry()
        sr.register(ServiceInstance("api", "localhost", 8080))
        sr.unregister("api", "localhost", 8080)
        assert len(sr.discover("api")) == 0

    def test_heartbeat(self):
        sr = ServiceRegistry()
        sr.register(ServiceInstance("api", "localhost", 8080))
        assert sr.heartbeat("api", "localhost", 8080) is True
        assert sr.heartbeat("unknown", "x", 0) is False

    def test_healthy_instances(self):
        sr = ServiceRegistry()
        sr.register(ServiceInstance("api", "h1", 80))
        sr.register(ServiceInstance("api", "h2", 80))
        assert len(sr.healthy_instances("api")) == 2

    def test_discover_unknown(self):
        sr = ServiceRegistry()
        assert sr.discover("nonexistent") == []
