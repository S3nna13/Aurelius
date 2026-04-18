"""
Integration and unit tests for src.serving.api_server.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Dict

import pytest

from src.serving.api_server import (
    AureliusServer,
    ChatRequest,
    ChatResponse,
    create_server,
    make_mock_generate_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return an ephemeral port that is currently available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> AureliusServer:
    server = create_server("127.0.0.1", port, make_mock_generate_fn())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Wait until the port is accepting connections (up to 2 s).
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    return server


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _post(url: str, payload: Dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    _start_server(port)
    return f"http://127.0.0.1:{port}"


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
    port = _free_port()
    server = create_server("127.0.0.1", port, make_mock_generate_fn())
    assert isinstance(server, AureliusServer)
    server.server_close()


# ---------------------------------------------------------------------------
# Integration tests — HTTP
# ---------------------------------------------------------------------------

def test_health_returns_200(server_url):
    status, _ = _get(f"{server_url}/health")
    assert status == 200


def test_health_returns_status_key(server_url):
    _, body = _get(f"{server_url}/health")
    assert "status" in body
    assert body["status"] == "ok"


def test_models_returns_200(server_url):
    status, _ = _get(f"{server_url}/v1/models")
    assert status == 200


def test_chat_completions_returns_200(server_url):
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    status, _ = _post(f"{server_url}/v1/chat/completions", payload)
    assert status == 200


def test_chat_completions_has_choices_key(server_url):
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    _, body = _post(f"{server_url}/v1/chat/completions", payload)
    assert "choices" in body


def test_chat_completions_choices_has_message(server_url):
    payload = {
        "model": "aurelius",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    _, body = _post(f"{server_url}/v1/chat/completions", payload)
    assert len(body["choices"]) > 0
    assert "message" in body["choices"][0]


def test_chat_completions_missing_messages_returns_400(server_url):
    payload = {"model": "aurelius"}
    status, body = _post(f"{server_url}/v1/chat/completions", payload)
    assert status == 400
    assert "error" in body


def test_server_handles_multiple_sequential_requests(server_url):
    for i in range(5):
        payload = {
            "model": "aurelius",
            "messages": [{"role": "user", "content": f"Request {i}"}],
        }
        status, body = _post(f"{server_url}/v1/chat/completions", payload)
        assert status == 200
        assert "choices" in body
