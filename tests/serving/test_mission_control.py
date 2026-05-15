"""Tests for src.serving.mission_control."""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection

import pytest

from src.agent.command_dispatcher import CommandDispatcher
from src.agent.nl_command_parser import NLCommandParser
from src.serving.mission_control import (
    ActivityEntry,
    ActivityLog,
    create_mission_control_server,
)


@pytest.fixture
def server():
    srv = create_mission_control_server("127.0.0.1", 0)
    yield srv
    srv.server_close()


@pytest.fixture
def running_server():
    srv = create_mission_control_server("127.0.0.1", 0)
    srv.nl_parser = NLCommandParser()
    srv.command_dispatcher = CommandDispatcher()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield srv
    srv.shutdown()
    thread.join(timeout=2)
    srv.server_close()


def _request(srv, method, path, body=None):
    host, port = srv.server_address
    conn = HTTPConnection(host, port, timeout=5)
    headers = {"Content-Type": "application/json"} if body else {}
    data = json.dumps(body).encode("utf-8") if body else None
    conn.request(method, path, body=data, headers=headers)
    return conn.getresponse()


class TestActivityLog:
    def test_append_and_list(self) -> None:
        log = ActivityLog()
        entry = log.append("run skill x", True, "done")
        assert isinstance(entry, ActivityEntry)
        assert entry.command == "run skill x"
        assert entry.success is True
        assert entry.output == "done"
        assert len(log.list_entries()) == 1

    def test_limit(self) -> None:
        log = ActivityLog()
        for i in range(10):
            log.append(f"cmd {i}", True, "ok")
        entries = log.list_entries(limit=3)
        assert len(entries) == 3
        assert entries[-1].command == "cmd 9"

    def test_max_entries_prune(self) -> None:
        log = ActivityLog()
        log._max_entries = 5
        for i in range(10):
            log.append(f"cmd {i}", True, "ok")
        assert len(log.list_entries()) == 5

    def test_to_dicts(self) -> None:
        log = ActivityLog()
        log.append("test", False, "err")
        dicts = log.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["command"] == "test"
        assert dicts[0]["success"] is False


class TestGETIndex:
    def test_serves_html(self, running_server) -> None:
        resp = _request(running_server, "GET", "/")
        assert resp.status == 200
        body = resp.read()
        assert b"Aurelius Mission Control" in body
        assert b"text/html" in resp.getheader("Content-Type").encode()

    def test_index_html_alias(self, running_server) -> None:
        resp = _request(running_server, "GET", "/index.html")
        assert resp.status == 200
        assert b"Aurelius Mission Control" in resp.read()


class TestGETHealth:
    def test_health_ok(self, running_server) -> None:
        resp = _request(running_server, "GET", "/health")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ok"


class TestGETStatus:
    def test_status_structure(self, running_server) -> None:
        resp = _request(running_server, "GET", "/api/status")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "agents" in data
        assert "skills" in data
        assert "plugins" in data
        assert isinstance(data["agents"], list)
        assert isinstance(data["skills"], list)
        assert isinstance(data["plugins"], list)


class TestGETActivity:
    def test_activity_empty(self, running_server) -> None:
        resp = _request(running_server, "GET", "/api/activity")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "entries" in data
        assert data["entries"] == []


class TestPOSTCommand:
    def test_command_success(self, running_server) -> None:
        resp = _request(running_server, "POST", "/api/command", {"command": "hello"})
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["success"] is True
        assert "output" in data
        assert data["action"] == "chat"

    def test_command_run_skill(self, running_server) -> None:
        resp = _request(running_server, "POST", "/api/command", {"command": "list skills"})
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["action"] == "list_skills"

    def test_empty_command(self, running_server) -> None:
        resp = _request(running_server, "POST", "/api/command", {"command": ""})
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "error" in data

    def test_invalid_json(self, running_server) -> None:
        host, port = running_server.server_address
        import http.client

        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/api/command", body=b"not json", headers={})
        resp = conn.getresponse()
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "Invalid JSON" in data["error"]

    def test_command_not_string(self, running_server) -> None:
        resp = _request(running_server, "POST", "/api/command", {"command": 123})
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "command must be a string" in data["error"]

    def test_logs_activity(self, running_server) -> None:
        _request(running_server, "POST", "/api/command", {"command": "hello"})
        entries = running_server.activity_log.list_entries()
        assert len(entries) >= 1
        assert entries[-1].command == "hello"

    def test_content_length_too_large(self, running_server) -> None:
        host, port = running_server.server_address
        import http.client

        conn = http.client.HTTPConnection(host, port, timeout=5)
        body = b'{"command":"x"}'
        conn.request(
            "POST",
            "/api/command",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(2 * 1024 * 1024),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 413


class Test404:
    def test_unknown_get(self, running_server) -> None:
        resp = _request(running_server, "GET", "/unknown")
        assert resp.status == 404

    def test_unknown_post(self, running_server) -> None:
        resp = _request(running_server, "POST", "/unknown", {})
        assert resp.status == 404


class TestServerConstructor:
    def test_defaults(self) -> None:
        srv = create_mission_control_server("127.0.0.1", 0, bind_and_activate=False)
        assert srv.nl_parser is None
        assert srv.command_dispatcher is None
        assert srv.activity_log is not None
        srv.server_close()

    def test_with_parser_and_dispatcher(self) -> None:
        parser = NLCommandParser()
        dispatcher = CommandDispatcher()
        srv = create_mission_control_server(
            "127.0.0.1",
            0,
            nl_parser=parser,
            command_dispatcher=dispatcher,
            bind_and_activate=False,
        )
        assert srv.nl_parser is parser
        assert srv.command_dispatcher is dispatcher
        srv.server_close()
