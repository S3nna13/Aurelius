"""Tests for heartbeat monitor."""
from __future__ import annotations

import time

import pytest

from src.protocol.heartbeat import HeartbeatMonitor


class TestHeartbeatMonitor:
    def test_beat_and_is_alive(self):
        hm = HeartbeatMonitor(timeout=10.0)
        hm.beat("svc1")
        assert hm.is_alive("svc1") is True

    def test_unknown_service_not_alive(self):
        hm = HeartbeatMonitor()
        assert hm.is_alive("unknown") is False

    def test_stale_service_detected(self):
        hm = HeartbeatMonitor(timeout=0)
        hm.beat("svc1")
        time.sleep(0.01)
        assert hm.is_alive("svc1") is False

    def test_active_services(self):
        hm = HeartbeatMonitor(timeout=10.0)
        hm.beat("a")
        hm.beat("b")
        assert sorted(hm.active_services()) == ["a", "b"]

    def test_dead_services(self):
        hm = HeartbeatMonitor(timeout=0)
        hm.beat("dead1")
        time.sleep(0.01)
        assert "dead1" in hm.dead_services()