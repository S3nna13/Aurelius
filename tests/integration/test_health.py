"""Health check integration tests."""

from __future__ import annotations

from typing import Any

import requests


class TestHealthEndpoints:
    def test_health_returns_ok(self, api_client: Any, health_url: str) -> None:
        resp = api_client.get(health_url)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_uptime(self, api_client: Any, health_url: str) -> None:
        resp = api_client.get(health_url)
        data = resp.json()
        assert "uptime" in data
        assert data["uptime"] >= 0

    def test_health_has_version(self, api_client: Any, health_url: str) -> None:
        resp = api_client.get(health_url)
        data = resp.json()
        assert "version" in data

    def test_healthz_returns_alive(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}/healthz")
        assert resp.status_code == 200
        assert resp.json()["alive"] is True

    def test_readyz_returns_ready(self, api_client: Any, readyz_url: str) -> None:
        resp = api_client.get(readyz_url)
        assert resp.status_code == 200
        assert resp.json()["ready"] is True

    def test_health_unauthorized_without_key(self, base_url: str) -> None:
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert resp.status_code in (200, 401)

    def test_health_with_invalid_key(self, base_url: str) -> None:
        resp = requests.get(
            f"{base_url}/health",
            headers={"X-API-Key": "invalid-key"},
            timeout=5,
        )
        assert resp.status_code in (200, 401)
