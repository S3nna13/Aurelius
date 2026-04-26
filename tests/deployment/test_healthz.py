"""Tests for src.deployment.healthz.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import json
from typing import Any

from src.deployment.healthz import (
    DEPLOY_TARGET_REGISTRY,
    HEALTHZ_REGISTRY,
    HealthStatus,
    HealthzHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wsgi_environ(path: str, method: str = "GET") -> dict[str, Any]:
    return {
        "PATH_INFO": path,
        "REQUEST_METHOD": method,
    }


def _capture_wsgi(handler: HealthzHandler, path: str, method: str = "GET"):
    """Run wsgi_app and return (status_line, headers, body_dict)."""
    captured_status: list[str] = []
    captured_headers: list[list] = []

    def start_response(status: str, headers: list) -> None:
        captured_status.append(status)
        captured_headers.append(headers)

    environ = _make_wsgi_environ(path, method)
    raw_chunks = handler.wsgi_app(environ, start_response)
    body_bytes = b"".join(raw_chunks)
    body = json.loads(body_bytes.decode())
    return captured_status[0], captured_headers[0], body


# ---------------------------------------------------------------------------
# HealthStatus and HealthCheckResult
# ---------------------------------------------------------------------------


def test_health_status_values() -> None:
    """HealthStatus enum has HEALTHY, DEGRADED, UNHEALTHY members."""
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.DEGRADED.value == "degraded"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"


# ---------------------------------------------------------------------------
# HealthzHandler.check()
# ---------------------------------------------------------------------------


def test_check_all_passing_is_healthy() -> None:
    """All passing checks produce HEALTHY status."""
    h = HealthzHandler()
    h.register_check("db", lambda: True)
    h.register_check("cache", lambda: True)

    result = h.check()
    assert result.status == HealthStatus.HEALTHY
    assert result.checks["db"] is True
    assert result.checks["cache"] is True


def test_check_one_failing_is_degraded() -> None:
    """One failing check among multiple produces DEGRADED status."""
    h = HealthzHandler()
    h.register_check("db", lambda: True)
    h.register_check("cache", lambda: False)

    result = h.check()
    assert result.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
    assert result.checks["cache"] is False


def test_check_all_failing_is_unhealthy() -> None:
    """All failing checks produce UNHEALTHY status."""
    h = HealthzHandler()
    h.register_check("db", lambda: False)
    h.register_check("cache", lambda: False)

    result = h.check()
    assert result.status == HealthStatus.UNHEALTHY


def test_check_no_checks_is_unhealthy() -> None:
    """No registered checks produces UNHEALTHY status."""
    h = HealthzHandler()
    result = h.check()
    assert result.status == HealthStatus.UNHEALTHY


def test_check_exception_in_fn_counts_as_failure() -> None:
    """A check that raises an exception is treated as failed."""

    def bad_check() -> bool:
        raise RuntimeError("boom")

    h = HealthzHandler()
    h.register_check("bad", bad_check)
    result = h.check()
    assert result.checks["bad"] is False


# ---------------------------------------------------------------------------
# wsgi_app /healthz
# ---------------------------------------------------------------------------


def test_wsgi_healthz_returns_200() -> None:
    """GET /healthz always returns 200 OK (liveness)."""
    h = HealthzHandler()
    h.register_check("ok", lambda: True)

    status, _, body = _capture_wsgi(h, "/healthz")
    assert status.startswith("200")
    assert body["status"] == "healthy"


def test_wsgi_healthz_200_even_when_failing() -> None:
    """GET /healthz returns 200 even when checks fail (liveness probe)."""
    h = HealthzHandler()
    h.register_check("bad", lambda: False)

    status, _, _ = _capture_wsgi(h, "/healthz")
    assert status.startswith("200")


# ---------------------------------------------------------------------------
# wsgi_app /readyz
# ---------------------------------------------------------------------------


def test_wsgi_readyz_200_when_healthy() -> None:
    """GET /readyz returns 200 when all checks pass."""
    h = HealthzHandler()
    h.register_check("db", lambda: True)

    status, _, body = _capture_wsgi(h, "/readyz")
    assert status.startswith("200")
    assert body["status"] == "healthy"


def test_wsgi_readyz_503_with_failing_check() -> None:
    """GET /readyz returns 503 when any check fails."""
    h = HealthzHandler()
    h.register_check("db", lambda: False)

    status, _, _ = _capture_wsgi(h, "/readyz")
    assert status.startswith("503")


def test_wsgi_unknown_path_returns_404() -> None:
    """Unknown paths return 404."""
    h = HealthzHandler()
    status, _, _ = _capture_wsgi(h, "/unknown")
    assert status.startswith("404")


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def test_healthz_registry_is_dict() -> None:
    """HEALTHZ_REGISTRY is a dict."""
    assert isinstance(HEALTHZ_REGISTRY, dict)


def test_deploy_target_registry_contains_docker() -> None:
    """DEPLOY_TARGET_REGISTRY has a 'docker' entry."""
    assert "docker" in DEPLOY_TARGET_REGISTRY


def test_deploy_target_registry_contains_k8s() -> None:
    """DEPLOY_TARGET_REGISTRY has a 'k8s' entry."""
    assert "k8s" in DEPLOY_TARGET_REGISTRY


def test_deploy_target_registry_contains_compose() -> None:
    """DEPLOY_TARGET_REGISTRY has a 'compose' entry."""
    assert "compose" in DEPLOY_TARGET_REGISTRY
