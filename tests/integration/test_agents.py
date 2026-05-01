"""Agent integration tests."""

from __future__ import annotations

from typing import Any

import pytest


class TestAgents:
    AGENTS_URL = "/api/agents"

    def test_list_agents(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

    def test_get_agent_by_id(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        agents = resp.json().get("agents", [])
        if not agents:
            pytest.skip("No agents available")
        agent_id = agents[0]["id"]
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}/{agent_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == agent_id

    def test_get_nonexistent_agent(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}/nonexistent")
        assert resp.status_code == 404

    def test_agent_has_state(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        agents = resp.json().get("agents", [])
        if agents:
            assert "state" in agents[0]

    def test_agent_has_role(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        agents = resp.json().get("agents", [])
        if agents:
            assert "role" in agents[0]

    def test_send_command_to_agent(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        agents = resp.json().get("agents", [])
        if not agents:
            pytest.skip("No agents available")
        agent_id = agents[0]["id"]
        resp = api_client.post(
            f"{base_url}{self.AGENTS_URL}/{agent_id}/command",
            json={"command": "test.ping"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_change_agent_state(self, api_client: Any, base_url: str) -> None:
        resp = api_client.get(f"{base_url}{self.AGENTS_URL}")
        agents = resp.json().get("agents", [])
        if not agents:
            pytest.skip("No agents available")
        agent_id = agents[0]["id"]
        resp = api_client.post(
            f"{base_url}{self.AGENTS_URL}/{agent_id}/state",
            json={"state": "idle"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
