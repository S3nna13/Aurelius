"""Deployment health probe with stdlib-only checks.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass
class ProbeResult:
    """Result of a single health probe."""

    status: str
    latency_ms: float
    status_code: int | None = None
    details: str | None = None


class HealthProbe:
    """Stdlib-only deployment health checker."""

    @staticmethod
    def _validate_url(url: str) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError("URL must be a non-empty string.")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL scheme must be http or https, got: {parsed.scheme}")
        if not parsed.netloc:
            raise ValueError("URL must contain a valid host.")

    @classmethod
    def check_http(cls, url: str, timeout: float = 5.0) -> dict[str, Any]:
        """Probe an HTTP/HTTPS endpoint.

        Args:
            url: Target URL (http or https only).
            timeout: Request timeout in seconds.

        Returns:
            dict with keys ``status``, ``latency_ms``, ``status_code``.

        Raises:
            ValueError: If the URL is empty or uses a disallowed scheme.
        """
        cls._validate_url(url)

        start = time.perf_counter()
        try:
            with urlopen(url, timeout=timeout) as response:  # noqa: S310  # nosec
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "status_code": response.status,
                }
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "status_code": None,
                "details": str(exc),
            }

    @classmethod
    def check_tcp(cls, host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
        """Probe a TCP host/port.

        Args:
            host: Hostname or IP address.
            port: TCP port number.
            timeout: Connection timeout in seconds.

        Returns:
            dict with keys ``status``, ``latency_ms``, ``status_code``.

        Raises:
            ValueError: If ``port`` is outside the valid 1–65535 range.
        """
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(f"Port must be an integer in [1, 65535], got: {port!r}")

        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "status_code": None,
                }
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "status_code": None,
                "details": str(exc),
            }

    @classmethod
    def check_process(cls, pid: int) -> dict[str, Any]:
        """Check whether a process exists and is accessible.

        Args:
            pid: Process ID to check.

        Returns:
            dict with keys ``status``, ``latency_ms``, ``status_code``.

        Raises:
            ValueError: If ``pid`` is not a positive integer.
        """
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"PID must be a positive integer, got: {pid!r}")

        start = time.perf_counter()
        try:
            os.kill(pid, 0)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "status_code": None,
            }
        except OSError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "status_code": None,
                "details": str(exc),
            }

    @classmethod
    def check_disk_usage(cls, path: str, threshold_percent: float = 90.0) -> dict[str, Any]:
        """Check disk usage at ``path`` against a threshold.

        Args:
            path: Filesystem path to inspect.
            threshold_percent: Usage percentage above which the check is unhealthy.

        Returns:
            dict with keys ``status``, ``latency_ms``, ``status_code``,
            and ``usage_percent``.

        Raises:
            ValueError: If ``threshold_percent`` is not positive.
        """
        if threshold_percent <= 0:
            raise ValueError(f"threshold_percent must be > 0, got: {threshold_percent}")

        start = time.perf_counter()
        try:
            usage = shutil.disk_usage(path)
            used_percent = (usage.used / usage.total) * 100.0
            latency_ms = (time.perf_counter() - start) * 1000
            status = "unhealthy" if used_percent >= threshold_percent else "healthy"
            return {
                "status": status,
                "latency_ms": latency_ms,
                "status_code": None,
                "usage_percent": used_percent,
            }
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "status_code": None,
                "details": str(exc),
            }
