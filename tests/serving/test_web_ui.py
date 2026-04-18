"""
Tests for src.serving.web_ui.
"""

import json
import socket
import threading
import urllib.request

import pytest

from src.serving.web_ui import (
    HTML_TEMPLATE,
    WebUIServer,
    create_ui_server,
    make_mock_generate_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> WebUIServer:
    server = create_ui_server("127.0.0.1", port, make_mock_generate_fn())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            import time
            time.sleep(0.05)
    return server


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    srv = _start_server(port)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


# ---------------------------------------------------------------------------
# HTML_TEMPLATE tests
# ---------------------------------------------------------------------------

def test_html_template_is_nonempty_string():
    assert isinstance(HTML_TEMPLATE, str)
    assert len(HTML_TEMPLATE) > 0


def test_html_template_contains_html_tag():
    assert "<html" in HTML_TEMPLATE


def test_html_template_contains_aurelius():
    assert "Aurelius" in HTML_TEMPLATE


def test_html_template_contains_fetch():
    assert "fetch(" in HTML_TEMPLATE


# ---------------------------------------------------------------------------
# make_mock_generate_fn tests
# ---------------------------------------------------------------------------

def test_mock_generate_fn_returns_callable():
    fn = make_mock_generate_fn()
    assert callable(fn)


def test_mock_generate_fn_output_contains_input():
    fn = make_mock_generate_fn()
    result = fn("hello world", [])
    assert "hello world" in result


# ---------------------------------------------------------------------------
# create_ui_server tests
# ---------------------------------------------------------------------------

def test_create_ui_server_returns_web_ui_server():
    port = _free_port()
    srv = create_ui_server("127.0.0.1", port, make_mock_generate_fn())
    assert isinstance(srv, WebUIServer)
    srv.server_close()


# ---------------------------------------------------------------------------
# HTTP integration tests
# ---------------------------------------------------------------------------

def test_get_root_returns_200(server_url):
    with urllib.request.urlopen(f"{server_url}/") as resp:
        assert resp.status == 200


def test_get_root_content_type_is_html(server_url):
    with urllib.request.urlopen(f"{server_url}/") as resp:
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type


def test_get_health_returns_200(server_url):
    with urllib.request.urlopen(f"{server_url}/health") as resp:
        assert resp.status == 200


def test_post_api_chat_returns_200(server_url):
    payload = json.dumps({"message": "hello", "history": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200


def test_post_api_chat_response_has_response_key(server_url):
    payload = json.dumps({"message": "ping", "history": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    assert "response" in data


def test_post_api_chat_response_value_is_string(server_url):
    payload = json.dumps({"message": "test message", "history": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    assert isinstance(data["response"], str)
