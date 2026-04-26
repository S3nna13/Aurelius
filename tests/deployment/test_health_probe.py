"""Tests for src.deployment.health_probe.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from src.deployment.health_probe import HealthProbe, ProbeResult

# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------


class TestProbeResult:
    def test_defaults(self):
        pr = ProbeResult(status="healthy", latency_ms=1.0)
        assert pr.status == "healthy"
        assert pr.latency_ms == 1.0
        assert pr.status_code is None
        assert pr.details is None

    def test_all_fields(self):
        pr = ProbeResult(status="unhealthy", latency_ms=2.0, status_code=503, details="err")
        assert pr.status_code == 503
        assert pr.details == "err"


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


class TestUrlValidation:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_http("")

    def test_non_string_url_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_http(None)  # type: ignore[arg-type]

    def test_file_scheme_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_http("file:///etc/passwd")

    def test_javascript_scheme_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_http("javascript:alert(1)")

    def test_missing_host_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_http("http://")

    def test_https_is_allowed(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        with patch("src.deployment.health_probe.urlopen", return_value=mock_response):
            result = HealthProbe.check_http("https://example.com/healthz")
            assert result["status"] == "healthy"
            assert result["status_code"] == 200


# ---------------------------------------------------------------------------
# HTTP check
# ---------------------------------------------------------------------------


class TestCheckHttp:
    def test_success_returns_healthy(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        with patch("src.deployment.health_probe.urlopen", return_value=mock_response):
            result = HealthProbe.check_http("http://localhost:8080/healthz")

        assert result["status"] == "healthy"
        assert result["status_code"] == 200
        assert result["latency_ms"] >= 0.0

    def test_failure_returns_unhealthy(self):
        with patch(
            "src.deployment.health_probe.urlopen",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = HealthProbe.check_http("http://localhost:9999/healthz")

        assert result["status"] == "unhealthy"
        assert result["status_code"] is None
        assert "refused" in result.get("details", "")

    def test_latency_is_populated_on_failure(self):
        with patch("src.deployment.health_probe.urlopen", side_effect=TimeoutError("slow")):
            result = HealthProbe.check_http("http://localhost:9999/healthz")

        assert result["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# TCP check
# ---------------------------------------------------------------------------


class TestCheckTcp:
    def test_success_returns_healthy(self):
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        with patch("src.deployment.health_probe.socket.create_connection", return_value=mock_sock):
            result = HealthProbe.check_tcp("127.0.0.1", 8080)

        assert result["status"] == "healthy"
        assert result["status_code"] is None
        assert result["latency_ms"] >= 0.0

    def test_failure_returns_unhealthy(self):
        with patch(
            "src.deployment.health_probe.socket.create_connection",
            side_effect=TimeoutError("timed out"),
        ):
            result = HealthProbe.check_tcp("192.0.2.1", 12345)

        assert result["status"] == "unhealthy"
        assert result["status_code"] is None
        assert "timed out" in result.get("details", "")

    def test_invalid_port_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_tcp("localhost", 0)

        with pytest.raises(ValueError):
            HealthProbe.check_tcp("localhost", 70000)

    def test_non_int_port_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_tcp("localhost", "8080")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Process check
# ---------------------------------------------------------------------------


class TestCheckProcess:
    def test_running_process_returns_healthy(self):
        with patch("src.deployment.health_probe.os.kill", return_value=None):
            result = HealthProbe.check_process(os.getpid())

        assert result["status"] == "healthy"
        assert result["latency_ms"] >= 0.0

    def test_missing_process_returns_unhealthy(self):
        with patch("src.deployment.health_probe.os.kill", side_effect=OSError("No such process")):
            result = HealthProbe.check_process(999999)

        assert result["status"] == "unhealthy"
        assert "No such process" in result.get("details", "")

    def test_invalid_pid_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_process(0)

        with pytest.raises(ValueError):
            HealthProbe.check_process(-5)

    def test_non_int_pid_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_process("1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Disk usage check
# ---------------------------------------------------------------------------


class TestCheckDiskUsage:
    def test_healthy_when_below_threshold(self):
        mock_usage = shutil._ntuple_diskusage(total=100, used=40, free=60)
        with patch("src.deployment.health_probe.shutil.disk_usage", return_value=mock_usage):
            result = HealthProbe.check_disk_usage("/", threshold_percent=90.0)

        assert result["status"] == "healthy"
        assert result["usage_percent"] == 40.0
        assert result["latency_ms"] >= 0.0

    def test_unhealthy_when_at_threshold(self):
        mock_usage = shutil._ntuple_diskusage(total=100, used=90, free=10)
        with patch("src.deployment.health_probe.shutil.disk_usage", return_value=mock_usage):
            result = HealthProbe.check_disk_usage("/", threshold_percent=90.0)

        assert result["status"] == "unhealthy"
        assert result["usage_percent"] == 90.0

    def test_unhealthy_when_above_threshold(self):
        mock_usage = shutil._ntuple_diskusage(total=100, used=95, free=5)
        with patch("src.deployment.health_probe.shutil.disk_usage", return_value=mock_usage):
            result = HealthProbe.check_disk_usage("/", threshold_percent=90.0)

        assert result["status"] == "unhealthy"
        assert result["usage_percent"] == 95.0

    def test_failure_returns_unhealthy(self):
        with patch(
            "src.deployment.health_probe.shutil.disk_usage",
            side_effect=FileNotFoundError("missing"),
        ):
            result = HealthProbe.check_disk_usage("/nonexistent", threshold_percent=90.0)

        assert result["status"] == "unhealthy"
        assert "missing" in result.get("details", "")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            HealthProbe.check_disk_usage("/", threshold_percent=0.0)

        with pytest.raises(ValueError):
            HealthProbe.check_disk_usage("/", threshold_percent=-10.0)
