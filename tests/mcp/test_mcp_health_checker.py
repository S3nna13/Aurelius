"""Tests for mcp_health_checker."""
from __future__ import annotations
from src.mcp.mcp_health_checker import McpHealthChecker, HealthStatus
class TestHealthStatus:
    def test_values(self): assert HealthStatus.HEALTHY.value=="healthy"; assert HealthStatus.DEGRADED.value=="degraded"
class TestMcpHealthChecker:
    def test_initial_healthy(self): h=McpHealthChecker("test-server"); h.ping(); assert h.status()==HealthStatus.HEALTHY
    def test_timeout_causes_unhealthy(self): h=McpHealthChecker("slow-server",timeout=0.5); h.ping(); h._last_latency=999.0; assert h.status()==HealthStatus.UNHEALTHY
    def test_stats_track_calls(self): h=McpHealthChecker("s"); h.ping();h.ping();s=h.get_stats();assert s["total_pings"]==2
    def test_reset(self): h=McpHealthChecker("s");h.ping();h.reset();assert h.get_stats()["total_pings"]==0
