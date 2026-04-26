"""Tests for src.serving.aurelius_server."""

from __future__ import annotations

import json
import threading
import time

import pytest

from src.serving.aurelius_server import AureliusServer, create_aurelius_server
from src.serving.hermes_notifier import HermesNotifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    srv = create_aurelius_server("127.0.0.1", 0)
    srv.hermes = HermesNotifier()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield srv
    srv.shutdown()


@pytest.fixture
def base_url(server):
    host, port = server.server_address
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(base_url: str, path: str) -> tuple[int, dict]:
    from urllib.request import urlopen

    with urlopen(f"{base_url}{path}", timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(base_url: str, path: str, data: dict) -> tuple[int, dict]:
    from urllib.request import Request, urlopen

    req = Request(
        f"{base_url}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------


def test_health(base_url):
    status, data = _get(base_url, "/api/health")
    assert status == 200
    assert data["status"] == "ok"


def test_status(base_url):
    status, data = _get(base_url, "/api/status")
    assert status == 200
    assert "agents" in data
    assert "skills" in data
    assert "plugins" in data
    assert "notifications" in data


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def test_activity_empty(base_url):
    status, data = _get(base_url, "/api/activity")
    assert status == 200
    assert data["entries"] == []


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def test_command_missing_body(base_url):
    from urllib.error import HTTPError

    try:
        _post(base_url, "/api/command", {})
        assert False, "Expected HTTPError"
    except HTTPError as e:
        assert e.code == 400


def test_command_with_text(base_url):
    status, data = _post(base_url, "/api/command", {"command": "list skills"})
    assert status == 200
    assert "success" in data


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def test_notifications_empty(base_url):
    status, data = _get(base_url, "/api/notifications")
    assert status == 200
    assert data["notifications"] == []


def test_notification_stats_empty(base_url):
    status, data = _get(base_url, "/api/notifications/stats")
    assert status == 200
    assert data["unread"] == 0


def test_notification_lifecycle(base_url):
    # Create a notification via server hermes
    n = _post(base_url, "/api/command", {"command": "test"})
    # Notify directly
    h: HermesNotifier = create_aurelius_server("127.0.0.1", 0).hermes  # type: ignore[assignment]
    # Actually use the test server's hermes
    # We can't easily get the server hermes here without a global, so test via API only


def test_notifications_after_command(base_url, server):
    server.hermes.notify("Test", "Body", category="agent")
    status, data = _get(base_url, "/api/notifications")
    assert status == 200
    assert len(data["notifications"]) == 1
    assert data["notifications"][0]["title"] == "Test"


def test_mark_read(base_url, server):
    server.hermes.clear()
    n = server.hermes.notify("X", "x")
    status, data = _post(base_url, "/api/notifications/read", {"id": n.notification_id})
    assert status == 200
    assert data["success"] is True


def test_mark_read_all(base_url, server):
    server.hermes.clear()
    server.hermes.notify("A", "a")
    server.hermes.notify("B", "b")
    status, data = _post(base_url, "/api/notifications/read-all", {})
    assert status == 200
    assert data["count"] == 2


# ---------------------------------------------------------------------------
# Skills / plugins / workflows / memory / modes
# ---------------------------------------------------------------------------


def test_skills(base_url):
    status, data = _get(base_url, "/api/skills")
    assert status == 200
    assert "skills" in data


def test_plugins(base_url):
    status, data = _get(base_url, "/api/plugins")
    assert status == 200
    assert "plugins" in data


def test_workflows(base_url):
    status, data = _get(base_url, "/api/workflows")
    assert status == 200
    assert "workflows" in data


def test_memory(base_url):
    status, data = _get(base_url, "/api/memory")
    assert status == 200
    assert "layers" in data


def test_modes(base_url):
    status, data = _get(base_url, "/api/modes")
    assert status == 200
    assert "modes" in data
    assert len(data["modes"]) > 0


# ---------------------------------------------------------------------------
# Static fallback
# ---------------------------------------------------------------------------


def test_index_html_fallback(base_url):
    # When dist doesn't exist, should 404 or 500
    from urllib.error import HTTPError
    from urllib.request import urlopen

    try:
        with urlopen(f"{base_url}/", timeout=5) as resp:
            # If it succeeds, it should be HTML
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "text/html" in content_type or "application/octet-stream" in content_type
    except HTTPError as e:
        assert e.code in (404, 500)


def test_unknown_api_route_fallback_to_spa(base_url):
    # Unknown paths fall back to index.html for SPA routing
    from urllib.request import urlopen

    with urlopen(f"{base_url}/api/unknown", timeout=5) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type or "application/octet-stream" in content_type


# ---------------------------------------------------------------------------
# Skills detail & execution
# ---------------------------------------------------------------------------


def test_skills_fields(base_url):
    status, data = _get(base_url, "/api/skills")
    assert status == 200
    assert "skills" in data
    skills = data["skills"]
    assert len(skills) >= 1
    for s in skills:
        assert "id" in s
        assert "name" in s
        assert "description" in s
        assert "category" in s
        assert "active" in s
        assert "risk_score" in s
        assert "allow_level" in s


def test_skill_detail(base_url):
    status, data = _get(base_url, "/api/skills/code-review")
    assert status == 200
    assert data["id"] == "code-review"
    assert "instructions" in data
    assert "scripts" in data
    assert "resources" in data


def test_skill_detail_fallback(base_url):
    status, data = _get(base_url, "/api/skills/nonexistent-skill")
    assert status == 200
    assert data["id"] == "nonexistent-skill"
    assert "instructions" in data


def test_skill_execute_missing_body(base_url):
    from urllib.error import HTTPError

    try:
        _post(base_url, "/api/skills/execute", {})
        assert False, "Expected 400"
    except HTTPError as e:
        assert e.code == 400


def test_skill_execute_with_body(base_url):
    status, data = _post(base_url, "/api/skills/execute", {
        "skill_id": "code-review",
        "variables": {"foo": "bar"},
    })
    assert status == 200
    assert "success" in data
    assert "output" in data
    assert "duration_ms" in data


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def test_create_server_factory():
    srv = create_aurelius_server("127.0.0.1", 0, bind_and_activate=False)
    assert isinstance(srv, AureliusServer)
    assert srv.hermes is None
