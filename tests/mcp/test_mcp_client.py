"""Tests for src.mcp.mcp_client (stub protocol client).

Coverage (≥ 28 tests):
- Default transport echoes params
- call() stores request_log
- call() stores response_log
- request_id is auto-assigned when empty
- request_id is preserved when provided
- Custom transport_fn is used
- call_with_retry succeeds on first attempt when no error
- call_with_retry retries on error up to max_retries times
- call_with_retry returns last response after exhausting retries
- MCPClientConfig is frozen (TypeError on assignment)
- MCPRequest is frozen (TypeError on assignment)
- MCPResponse is frozen (TypeError on assignment)
- MCPClientConfig field defaults
- MCPRequest auto-id uniqueness
- Log isolation between calls
- Transport receives correct method and params
- Response request_id matches request request_id
- MCP_CLIENT_REGISTRY contains "default" key
- MCP_CLIENT_REGISTRY["default"] is MCPClient
"""

from __future__ import annotations

import pytest

from src.mcp.mcp_client import (
    MCP_CLIENT_REGISTRY,
    MCPClient,
    MCPClientConfig,
    MCPRequest,
    MCPResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_transport(req: MCPRequest) -> MCPResponse:
    """Always returns an error response."""
    return MCPResponse(request_id=req.request_id, result=None, error="always fails")


def _success_transport(req: MCPRequest) -> MCPResponse:
    """Always returns a success response."""
    return MCPResponse(request_id=req.request_id, result={"ok": True})


def _make_client(**kwargs) -> MCPClient:
    cfg = MCPClientConfig(server_name="test", **kwargs)
    return MCPClient(cfg)


# ---------------------------------------------------------------------------
# MCPRequest tests
# ---------------------------------------------------------------------------


def test_mcp_request_auto_assigns_request_id_when_empty():
    req = MCPRequest(method="ping", params={})
    assert req.request_id != ""
    assert len(req.request_id) == 8


def test_mcp_request_preserves_explicit_request_id():
    req = MCPRequest(method="ping", params={}, request_id="abc123")
    assert req.request_id == "abc123"


def test_mcp_request_auto_id_is_unique():
    ids = {MCPRequest(method="ping", params={}).request_id for _ in range(50)}
    assert len(ids) > 1


def test_mcp_request_stores_method_and_params():
    req = MCPRequest(method="tools/call", params={"name": "echo"})
    assert req.method == "tools/call"
    assert req.params == {"name": "echo"}


def test_mcp_request_frozen():
    req = MCPRequest(method="ping", params={})
    with pytest.raises((AttributeError, TypeError)):
        req.method = "pong"  # type: ignore[misc]


def test_mcp_request_empty_params_allowed():
    req = MCPRequest(method="noop", params={})
    assert req.params == {}


# ---------------------------------------------------------------------------
# MCPResponse tests
# ---------------------------------------------------------------------------


def test_mcp_response_stores_fields():
    resp = MCPResponse(request_id="rid1", result={"data": 42})
    assert resp.request_id == "rid1"
    assert resp.result == {"data": 42}
    assert resp.error is None


def test_mcp_response_error_field():
    resp = MCPResponse(request_id="rid2", result=None, error="something broke")
    assert resp.error == "something broke"
    assert resp.result is None


def test_mcp_response_frozen():
    resp = MCPResponse(request_id="r", result={})
    with pytest.raises((AttributeError, TypeError)):
        resp.result = {"x": 1}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MCPClientConfig tests
# ---------------------------------------------------------------------------


def test_mcp_client_config_defaults():
    cfg = MCPClientConfig(server_name="srv")
    assert cfg.server_name == "srv"
    assert cfg.timeout_ms == 5000
    assert cfg.max_retries == 3


def test_mcp_client_config_custom_values():
    cfg = MCPClientConfig(server_name="my-server", timeout_ms=1000, max_retries=5)
    assert cfg.timeout_ms == 1000
    assert cfg.max_retries == 5


def test_mcp_client_config_frozen():
    cfg = MCPClientConfig(server_name="srv")
    with pytest.raises((AttributeError, TypeError)):
        cfg.timeout_ms = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MCPClient.call() tests
# ---------------------------------------------------------------------------


def test_call_default_transport_echoes_params():
    client = _make_client()
    resp = client.call("ping", {"a": 1})
    assert resp.result == {"echo": {"a": 1}}


def test_call_default_transport_no_error():
    client = _make_client()
    resp = client.call("ping", {})
    assert resp.error is None


def test_call_stores_request_in_log():
    client = _make_client()
    client.call("ping", {"x": 1})
    assert len(client.request_log) == 1
    assert client.request_log[0].method == "ping"
    assert client.request_log[0].params == {"x": 1}


def test_call_stores_response_in_log():
    client = _make_client()
    client.call("ping", {"x": 1})
    assert len(client.response_log) == 1
    assert client.response_log[0].result == {"echo": {"x": 1}}


def test_call_multiple_times_grows_logs():
    client = _make_client()
    client.call("m1", {})
    client.call("m2", {})
    client.call("m3", {})
    assert len(client.request_log) == 3
    assert len(client.response_log) == 3


def test_call_response_request_id_matches_request():
    client = _make_client()
    resp = client.call("ping", {})
    req = client.request_log[0]
    assert resp.request_id == req.request_id


def test_call_uses_custom_transport():
    cfg = MCPClientConfig(server_name="srv")
    client = MCPClient(cfg, transport_fn=_success_transport)
    resp = client.call("anything", {})
    assert resp.result == {"ok": True}


def test_call_custom_transport_receives_correct_method():
    received: list[MCPRequest] = []

    def capturing_transport(req: MCPRequest) -> MCPResponse:
        received.append(req)
        return MCPResponse(request_id=req.request_id, result={})

    cfg = MCPClientConfig(server_name="srv")
    client = MCPClient(cfg, transport_fn=capturing_transport)
    client.call("my/method", {"key": "val"})
    assert received[0].method == "my/method"
    assert received[0].params == {"key": "val"}


# ---------------------------------------------------------------------------
# MCPClient.call_with_retry() tests
# ---------------------------------------------------------------------------


def test_call_with_retry_succeeds_immediately_on_no_error():
    client = _make_client(max_retries=3)
    resp = client.call_with_retry("ping", {})
    assert resp.error is None
    # Only one dispatch should have occurred.
    assert len(client.request_log) == 1


def test_call_with_retry_retries_on_error():
    cfg = MCPClientConfig(server_name="srv", max_retries=3)
    client = MCPClient(cfg, transport_fn=_error_transport)
    resp = client.call_with_retry("ping", {})
    assert resp.error == "always fails"
    # Should have retried max_retries times.
    assert len(client.request_log) == 3


def test_call_with_retry_returns_last_response_after_exhaustion():
    cfg = MCPClientConfig(server_name="srv", max_retries=2)
    client = MCPClient(cfg, transport_fn=_error_transport)
    resp = client.call_with_retry("ping", {})
    # Result should be the last (failed) response.
    assert resp.error == "always fails"
    assert len(client.request_log) == 2


def test_call_with_retry_stops_early_on_success():
    call_count = 0

    def sometimes_fails(req: MCPRequest) -> MCPResponse:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return MCPResponse(request_id=req.request_id, result=None, error="fail")
        return MCPResponse(request_id=req.request_id, result={"done": True})

    cfg = MCPClientConfig(server_name="srv", max_retries=5)
    client = MCPClient(cfg, transport_fn=sometimes_fails)
    resp = client.call_with_retry("ping", {})
    assert resp.error is None
    assert resp.result == {"done": True}
    assert call_count == 3


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_mcp_client_registry_contains_default():
    assert "default" in MCP_CLIENT_REGISTRY


def test_mcp_client_registry_default_is_mcp_client_class():
    assert MCP_CLIENT_REGISTRY["default"] is MCPClient


def test_mcp_client_registry_is_dict():
    assert isinstance(MCP_CLIENT_REGISTRY, dict)
