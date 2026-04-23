"""Health check endpoints as a standalone stdlib WSGI application.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class HealthStatus(Enum):
    """Enumeration of health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Aggregated result of running all registered health checks."""

    status: HealthStatus
    message: str
    checks: dict[str, bool] = field(default_factory=dict)


class HealthzHandler:
    """Handles /healthz and /readyz endpoints via a minimal WSGI interface."""

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, fn: Callable[[], bool]) -> None:
        """Register a named health-check callable.

        Args:
            name: Unique identifier for this check.
            fn: Callable that returns True if healthy, False otherwise.

        Raises:
            ValueError: If name is empty or fn is not callable.
        """
        if not name:
            raise ValueError("Health check name must not be empty.")
        if not callable(fn):
            raise ValueError(f"Health check '{name}' must be callable.")
        self._checks[name] = fn

    def check(self) -> HealthCheckResult:
        """Run all registered checks and return aggregated result.

        - All passing → HEALTHY
        - Any failing → DEGRADED (if some pass) or UNHEALTHY (if none pass / no checks)
        """
        results: dict[str, bool] = {}
        for name, fn in self._checks.items():
            try:
                results[name] = bool(fn())
            except Exception:  # noqa: BLE001
                results[name] = False

        if not results:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="No health checks registered.",
                checks=results,
            )

        passing = sum(1 for v in results.values() if v)
        total = len(results)

        if passing == total:
            status = HealthStatus.HEALTHY
            message = "All checks passed."
        elif passing > 0:
            status = HealthStatus.DEGRADED
            message = f"{passing}/{total} checks passed."
        else:
            status = HealthStatus.UNHEALTHY
            message = "All checks failed."

        return HealthCheckResult(status=status, message=message, checks=results)

    def wsgi_app(self, environ: dict, start_response: Callable) -> list[bytes]:
        """Minimal WSGI handler for /healthz and /readyz.

        GET /healthz → 200 OK (always, for liveness)
        GET /readyz  → 200 OK if all checks pass, 503 otherwise

        Raises:
            ValueError: If the HTTP method is not GET (for these two paths).
        """
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path not in ("/healthz", "/readyz"):
            body = json.dumps({"error": "Not Found"}).encode()
            start_response("404 Not Found", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if method != "GET":
            body = json.dumps({"error": "Method Not Allowed"}).encode()
            start_response("405 Method Not Allowed", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        result = self.check()
        payload = {
            "status": result.status.value,
            "message": result.message,
            "checks": result.checks,
        }
        body = json.dumps(payload).encode()

        if path == "/healthz":
            # Liveness — always 200
            status_line = "200 OK"
        else:
            # Readiness — 503 if not fully healthy
            if result.status == HealthStatus.HEALTHY:
                status_line = "200 OK"
            else:
                status_line = "503 Service Unavailable"

        start_response(status_line, [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ])
        return [body]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

HEALTHZ_REGISTRY: dict[str, HealthzHandler] = {}

DEPLOY_TARGET_REGISTRY: dict[str, dict] = {
    "docker": {
        "type": "docker",
        "description": "Build and run via Docker Engine",
        "build_context": ".",
        "dockerfile": "deployment/Dockerfile",
        "default_port": 8080,
        "healthcheck_path": "/healthz",
    },
    "k8s": {
        "type": "kubernetes",
        "description": "Deploy to Kubernetes cluster",
        "image_pull_policy": "IfNotPresent",
        "replicas": 1,
        "liveness_probe": "/healthz",
        "readiness_probe": "/readyz",
        "default_port": 8080,
    },
    "compose": {
        "type": "docker-compose",
        "description": "Local dev via docker-compose",
        "compose_file": "deployment/compose.yaml",
        "default_port": 8080,
        "healthcheck_path": "/healthz",
    },
}
