"""Search integration tests."""

from __future__ import annotations

from typing import Any


class TestSearch:
    URL = "/api/search"

    def test_search_requires_query(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}")
        assert resp.status_code == 400

    def test_search_returns_results(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}?q=system")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "query" in data

    def test_search_by_type(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}?q=system&type=activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "activity"

    def test_search_suggestions(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0

    def test_search_with_limit(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}?q=system&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
