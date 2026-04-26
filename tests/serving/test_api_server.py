"""Integration and unit tests for src.serving.api_server."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from src.serving.api_server import (
    AureliusRequestHandler,
    AureliusServer,
    ChatRequest,
    ChatResponse,
    create_server,
    make_mock_generate_fn,
)
from src.serving.auth_middleware import AuthConfig, AuthMiddleware
from src.serving.rate_limiter import RateLimitConfig, TokenBucketLimiter


class _NoCloseBytesIO(io.BytesIO):
    def close(self) -> None:  # pragma: no cover - BytesIO close is harmless
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
    auth_middleware=None,
    rate_limiter=None,
    extra_headers: dict | None = None,
):
    body = b""
    headers = [
        f"{method} {path} HTTP/1.1",
        "Host: localhost",
        "Connection: close",
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            headers.append(f"{k}: {v}")
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.append(f"Content-Length: {len(body)}")
        headers.append("Content-Type: application/json")
    request_bytes = ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body

    socket = _MemorySocket(request_bytes)
    server = SimpleNamespace(
        generate_fn=generate_fn or make_mock_generate_fn(),
        auth_middleware=auth_middleware,
        rate_limiter=rate_limiter,
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
# Unit tests — dataclasses and helpers
# ---------------------------------------------------------------------------


def test_chat_request_instantiates():
    req = ChatRequest(model="aurelius", messages=[{"role": "user", "content": "Hi"}])
    assert req.model == "aurelius"
    assert req.messages[0]["role"] == "user"
    assert req.temperature == 0.7
    assert req.max_tokens == 512
    assert req.stream is False
    assert req.system is None


def test_chat_response_instantiates():
    resp = ChatResponse(
        id="chatcmpl-abc",
        object="chat.completion",
        created=1234567890,
        model="aurelius",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    assert resp.id == "chatcmpl-abc"
    assert resp.choices[0]["message"]["role"] == "assistant"


def test_make_mock_generate_fn_returns_callable():
    fn = make_mock_generate_fn()
    assert callable(fn)


def test_mock_generate_fn_returns_string():
    fn = make_mock_generate_fn()
    req = ChatRequest(model="aurelius", messages=[{"role": "user", "content": "Hello"}])
    result = fn(req)
    assert isinstance(result, str)
    assert "Hello" in result


def test_create_server_returns_aurelius_server():
    server = create_server(
        "127.0.0.1",
        0,
        make_mock_generate_fn(),
        bind_and_activate=False,
    )
    assert isinstance(server, AureliusServer)
    server.server_close()


# ---------------------------------------------------------------------------
# Integration tests — handler logic without real sockets
# ---------------------------------------------------------------------------


def test_health_returns_200():
    status, headers, body = _invoke(AureliusRequestHandler, "GET", "/health")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["status"] == "ok"


def test_health_returns_status_key():
    _, _, body = _invoke(AureliusRequestHandler, "GET", "/health")
    assert json.loads(body)["status"] == "ok"


def test_models_returns_200():
    status, headers, body = _invoke(AureliusRequestHandler, "GET", "/v1/models")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["object"] == "list"


def test_chat_completions_returns_200():
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    status, headers, body = _invoke(
        AureliusRequestHandler,
        "POST",
        "/v1/chat/completions",
        payload=payload,
    )
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_has_choices_key():
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    _, _, body = _invoke(
        AureliusRequestHandler,
        "POST",
        "/v1/chat/completions",
        payload=payload,
    )
    assert "choices" in json.loads(body)


def test_chat_completions_choices_has_message():
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    _, _, body = _invoke(
        AureliusRequestHandler,
        "POST",
        "/v1/chat/completions",
        payload=payload,
    )
    data = json.loads(body)
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]


def test_chat_completions_missing_messages_returns_400():
    payload = {"model": "aurelius"}
    status, headers, body = _invoke(
        AureliusRequestHandler,
        "POST",
        "/v1/chat/completions",
        payload=payload,
    )
    assert status == 400
    assert headers["Content-Type"] == "application/json"
    assert "error" in json.loads(body)


def test_server_handles_multiple_sequential_requests():
    for i in range(5):
        payload = {
            "model": "aurelius",
            "messages": [{"role": "user", "content": f"Request {i}"}],
        }
        status, headers, body = _invoke(
            AureliusRequestHandler,
            "POST",
            "/v1/chat/completions",
            payload=payload,
        )
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert "choices" in json.loads(body)


# ---------------------------------------------------------------------------
# Auth middleware wiring tests
# ---------------------------------------------------------------------------


def _auth_mw(key: str | None = None) -> AuthMiddleware:
    mw = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
    if key is not None:
        mw.add_key("test-key", key, frozenset({"read"}))
    return mw


def test_auth_rejects_missing_credentials():
    mw = _auth_mw("secret")
    status, _, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/v1/models",
        auth_middleware=mw,
    )
    assert status == 401
    error_payload = json.loads(body)["error"]
    assert "message" in error_payload
    assert "credential" in error_payload["message"].lower()


def test_auth_allows_valid_bearer_token():
    mw = _auth_mw("secret")
    status, _, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/v1/models",
        auth_middleware=mw,
        extra_headers={"Authorization": "Bearer secret"},
    )
    assert status == 200
    assert json.loads(body)["object"] == "list"


def test_auth_allows_valid_api_key_header():
    mw = _auth_mw("secret")
    status, _, body = _invoke(
        AureliusRequestHandler,
        "POST",
        "/v1/chat/completions",
        payload={"model": "aurelius", "messages": [{"role": "user", "content": "Hi"}]},
        auth_middleware=mw,
        extra_headers={"X-API-Key": "secret"},
    )
    assert status == 200
    assert "choices" in json.loads(body)


def test_health_skips_auth():
    mw = _auth_mw("secret")
    status, _, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/health",
        auth_middleware=mw,
    )
    assert status == 200
    assert json.loads(body)["status"] == "ok"


# ---------------------------------------------------------------------------
# Rate limiter wiring tests
# ---------------------------------------------------------------------------


def test_rate_limiter_rejects_when_exceeded():
    limiter = TokenBucketLimiter(
        RateLimitConfig(requests_per_second=0.0, burst_size=0)
    )
    status, headers, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/v1/models",
        rate_limiter=limiter,
    )
    assert status == 429
    assert "Retry-After" in headers
    assert "Rate limit exceeded" in json.loads(body)["error"]


def test_rate_limiter_allows_when_within_limit():
    limiter = TokenBucketLimiter(
        RateLimitConfig(requests_per_second=100.0, burst_size=10)
    )
    status, _, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/v1/models",
        rate_limiter=limiter,
    )
    assert status == 200
    assert json.loads(body)["object"] == "list"


def test_rate_limiter_uses_key_id_when_authenticated():
    mw = _auth_mw("secret")
    limiter = TokenBucketLimiter(
        RateLimitConfig(requests_per_second=0.0, burst_size=0)
    )
    status, _, body = _invoke(
        AureliusRequestHandler,
        "GET",
        "/v1/models",
        auth_middleware=mw,
        rate_limiter=limiter,
        extra_headers={"Authorization": "Bearer secret"},
    )
    assert status == 429
    assert "Rate limit exceeded" in json.loads(body)["error"]


def test_create_server_forwards_auth_and_rate_limiter():
    mw = _auth_mw()
    limiter = TokenBucketLimiter(
        RateLimitConfig(requests_per_second=1.0, burst_size=1)
    )
    server = create_server(
        "127.0.0.1",
        0,
        make_mock_generate_fn(),
        auth_middleware=mw,
        rate_limiter=limiter,
        bind_and_activate=False,
    )
    assert server.auth_middleware is mw
    assert server.rate_limiter is limiter
    server.server_close()
