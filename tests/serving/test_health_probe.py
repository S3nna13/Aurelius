"""Tests for health probe."""
from __future__ import annotations

import pytest

from src.serving.health_probe import (
    HealthProbe,
    HealthProbeType,
    HealthProbeResult,
)


class TestHealthProbe:
    """Test HealthProbe."""
    
    def test_liveness_passes(self):
        """Test liveness probe passes when callback returns True."""
        probe = HealthProbe(on_liveness=lambda: True)
        result = probe.check(HealthProbeType.LIVENESS)
        
        assert result.healthy is True
    
    def test_liveness_fails(self):
        """Test liveness probe fails when callback returns False."""
        probe = HealthProbe(on_liveness=lambda: False)
        result = probe.check(HealthProbeType.LIVENESS)
        
        assert result.healthy is False
    
    def test_readiness_requires_startup(self):
        """Test readiness requires startup to complete first."""
        probe = HealthProbe(
            on_startup=lambda: False,
            on_readiness=lambda: True,
        )
        result = probe.check(HealthProbeType.READINESS)
        
        assert result.healthy is False
    
    def test_readiness_after_startup(self):
        """Test readiness after startup completes."""
        probe = HealthProbe(
            on_startup=lambda: True,
            on_readiness=lambda: True,
        )
        probe.check(HealthProbeType.STARTUP)
        result = probe.check(HealthProbeType.READINESS)
        
        assert result.healthy is True
    
    def test_check_all(self):
        """Test checking all probes."""
        probe = HealthProbe()
        results = probe.check_all()
        
        assert len(results) == 3
        assert all(isinstance(r, HealthProbeResult) for r in results.values())
    
    def test_to_healthz_response_healthy(self):
        """Test healthy /healthz response."""
        probe = HealthProbe(on_startup=lambda: True)
        response = probe.to_healthz_response()
        
        assert "200" in response
        assert "liveness=1" in response
    
    def test_to_healthz_response_unhealthy(self):
        """Test unhealthy /healthz response."""
        probe = HealthProbe(on_startup=lambda: False)
        response = probe.to_healthz_response()
        
        assert "503" in response