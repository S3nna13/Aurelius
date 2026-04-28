"""Smoke tests for the agent cockpit helper."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.agent_cockpit import app


@pytest.fixture()
def client() -> TestClient:
    assert app is not None
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_workspace_session_flow(client: TestClient):
    workspace = client.post("/workspaces", json={"path": ""})
    assert workspace.status_code == 200
    workspace_data = workspace.json()
    assert workspace_data["path"]

    agents = client.get("/agents")
    assert agents.status_code == 200
    assert agents.json()

    session = client.post(
        f"/agents/{workspace_data['id']}/sessions",
        json={"agent_id": "coding", "workspace_id": workspace_data["id"]},
    )
    assert session.status_code == 200
    session_id = session.json()["id"]

    prompt = client.post(
        f"/sessions/{session_id}/prompt",
        json={"prompt": "Summarize the cockpit slice"},
    )
    assert prompt.status_code == 200
    result = prompt.json()["result"]
    assert result["mode"] == "coding"
    assert "Summarize" in result["prompt"]

