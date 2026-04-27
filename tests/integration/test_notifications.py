"""Notification integration tests."""

from __future__ import annotations

from typing import Any

import pytest


class TestNotifications:
    URL = "/api/notifications"

    def test_list_notifications(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}")
        assert resp.status_code == 200
        data = resp.json()
        assert "notifications" in data

    def test_get_stats(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "unread" in data
        assert "total" in data

    def test_create_notification(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(
            f"{base_url}{self.URL}",
            json={"title": "Test Notification", "body": "Test body", "priority": "high"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_mark_read(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}")
        notifs = resp.json().get("notifications", [])
        if not notifs:
            pytest.skip("No notifications available")
        nid = notifs[0]["id"]
        resp = api_client.post(f"{base_url}{self.URL}/{nid}/read")
        assert resp.status_code == 200

    def test_mark_all_read(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}/read-all", json={})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_create_notification_missing_title(self, api_client: Any, base_url: str) -> None:
        resp = api_client.post(f"{base_url}{self.URL}", json={})
        assert resp.status_code == 400

    def test_filter_by_category(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.URL}?category=startup")
        assert resp.status_code == 200
        data = resp.json()
        for n in data.get("notifications", []):
            assert n["category"] == "startup"
