"""Authentication integration tests."""

from __future__ import annotations

from typing import Any

import pytest


class TestAuth:
    URL = "/api/auth"

    def test_login_with_valid_key(self, api_client: Any, base_url: str, api_key: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}/login", json={"apiKey": api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "token" in data

    def test_login_with_invalid_key(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}/login", json={"apiKey": "invalid"})
        assert resp.status_code == 401

    def test_login_missing_key(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}/login", json={})
        assert resp.status_code == 400

    def test_register(self, api_client: Any, base_url: str) -> None:
        username = f"test-user-{__import__('time').time()}"
        resp = api_client.post(
            f"{base_url}{self.URL}/register",
            json={"username": username},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "apiKey" in data

    def test_register_duplicate(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(
            f"{base_url}{self.URL}/register",
            json={"username": "admin"},
        )
        assert resp.status_code == 409

    def test_list_users(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data

    def test_generate_api_key(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}/keys/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "apiKey" in data

    def test_list_api_keys(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert "keys" in data
