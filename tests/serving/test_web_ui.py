"""Tests for src.serving.web_ui."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from src.serving.web_ui import (
    HTML_TEMPLATE,
    WebUIHandler,
    WebUIServer,
    create_ui_server,
    make_mock_generate_fn,
)


class _NoCloseBytesIO(io.BytesIO):
    def close(self) -> None:  # pragma: no cover - harmless for in-memory tests
        pass


class _MemorySocket:
    def __init__(self, request_bytes: bytes) -> None:
        self._rfile = _NoCloseBytesIO(request_bytes)
        self._wfile = _NoCloseBytesIO()

    def makefile(self, mode: str, buffering: int | None = None):
        if "r" in mode:
            return self._rfile
        return self._wfile

    def close(self) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        self._wfile.write(data)

    def response_bytes(self) -> bytes:
        return self._wfile.getvalue()


def _request_bytes(
    method: str,
    path: str,
    payload: dict | None = None,
) -> bytes:
    body = b""
    headers = [
        f"{method} {path} HTTP/1.1",
        "Host: localhost",
        "Connection: close",
    ]
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.append(f"Content-Length: {len(body)}")
        headers.append("Content-Type: application/json")
    return ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body


def _invoke(
    handler_cls,
    method: str,
    path: str,
    payload: dict | None = None,
    generate_fn=None,
    api_url: str | None = None,
):
    socket = _MemorySocket(_request_bytes(method, path, payload))
    server = SimpleNamespace(
        generate_fn=generate_fn or make_mock_generate_fn(),
        api_url=api_url,
    )
    handler_cls(socket, ("127.0.0.1", 12345), server)
    raw = socket.response_bytes()
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line, *header_lines = head.decode("utf-8", errors="replace").split("\r\n")
    status = int(status_line.split()[1])
    headers = {}
    for line in header_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return status, headers, body


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
    srv = create_ui_server(
        "127.0.0.1",
        0,
        make_mock_generate_fn(),
        api_url="http://localhost:8080/v1/chat/completions",
        bind_and_activate=False,
    )
    assert isinstance(srv, WebUIServer)
    assert srv.api_url == "http://localhost:8080/v1/chat/completions"
    srv.server_close()


# ---------------------------------------------------------------------------
# Handler logic tests without real sockets
# ---------------------------------------------------------------------------


def test_get_root_returns_200():
    status, headers, body = _invoke(WebUIHandler, "GET", "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert body.decode("utf-8").startswith("<!DOCTYPE html>")


def test_get_root_content_type_is_html():
    _, headers, _ = _invoke(WebUIHandler, "GET", "/")
    assert "text/html" in headers["Content-Type"]


def test_get_health_returns_200():
    status, headers, body = _invoke(WebUIHandler, "GET", "/health")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["status"] == "ok"


def test_post_api_chat_returns_200():
    payload = {"message": "hello", "history": []}
    status, headers, body = _invoke(WebUIHandler, "POST", "/api/chat", payload=payload)
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert "response" in json.loads(body)


def test_post_api_chat_response_has_response_key():
    payload = {"message": "ping", "history": []}
    _, _, body = _invoke(WebUIHandler, "POST", "/api/chat", payload=payload)
    assert "response" in json.loads(body)


def test_post_api_chat_response_value_is_string():
    payload = {"message": "test message", "history": []}
    _, _, body = _invoke(WebUIHandler, "POST", "/api/chat", payload=payload)
    assert isinstance(json.loads(body)["response"], str)


def test_post_api_chat_proxies_to_configured_upstream(monkeypatch):
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "proxied response",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("src.serving.web_ui.urlopen", fake_urlopen)

    def unexpected_generate_fn(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("local generate_fn should not be used when api_url is set")

    payload = {
        "message": "hello",
        "history": [{"role": "assistant", "content": "prior reply"}],
    }
    status, headers, body = _invoke(
        WebUIHandler,
        "POST",
        "/api/chat",
        payload=payload,
        generate_fn=unexpected_generate_fn,
        api_url="http://example.test/v1/chat/completions",
    )

    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["response"] == "proxied response"
    assert captured["url"] == "http://example.test/v1/chat/completions"
    assert captured["timeout"] == 30
    assert captured["body"]["messages"] == [
        {"role": "assistant", "content": "prior reply"},
        {"role": "user", "content": "hello"},
    ]
