"""Tests for src.mcp.mcp_server.

Coverage:
- StdioMCPServer handles a valid JSON-RPC request payload correctly
- MCP_SERVER_REGISTRY contains the "stdio" entry
- list_mcp_servers() returns a non-empty list
- Malformed JSON payload does not crash; returns an error dict
- Missing required field is handled gracefully
"""

from __future__ import annotations

import io
import json

import pytest

from src.mcp.mcp_server import (
    MCP_SERVER_REGISTRY,
    MCPServerConfig,
    StdioMCPServer,
    get_mcp_server,
    list_mcp_servers,
    register_mcp_server,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_contains_stdio():
    assert "stdio" in MCP_SERVER_REGISTRY
    assert MCP_SERVER_REGISTRY["stdio"] is StdioMCPServer


def test_list_mcp_servers_non_empty():
    servers = list_mcp_servers()
    assert isinstance(servers, list)
    assert len(servers) >= 1
    assert "stdio" in servers


def test_get_mcp_server_returns_class():
    cls = get_mcp_server("stdio")
    assert cls is StdioMCPServer


def test_get_mcp_server_unknown_raises():
    with pytest.raises(KeyError, match="No MCP server registered"):
        get_mcp_server("__nonexistent__")


def test_register_mcp_server_type_check():
    with pytest.raises(TypeError):
        register_mcp_server("bad", object)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StdioMCPServer.handle_request tests
# ---------------------------------------------------------------------------


def _make_server() -> StdioMCPServer:
    return StdioMCPServer(MCPServerConfig(transport="stdio"))


def test_handle_initialize_request():
    server = _make_server()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    response = server.handle_request(payload)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "protocolVersion" in response["result"]
    assert "serverInfo" in response["result"]


def test_handle_tools_list_request():
    server = _make_server()
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    response = server.handle_request(payload)
    assert "result" in response
    assert "tools" in response["result"]
    assert isinstance(response["result"]["tools"], list)


def test_handle_unknown_method_returns_error():
    server = _make_server()
    payload = {"jsonrpc": "2.0", "id": 3, "method": "nonexistent/method", "params": {}}
    response = server.handle_request(payload)
    assert "error" in response
    assert response["error"]["code"] == -32601


def test_handle_missing_method_field():
    server = _make_server()
    payload = {"jsonrpc": "2.0", "id": 4}
    response = server.handle_request(payload)
    assert "error" in response
    assert response["error"]["code"] == -32600


def test_handle_wrong_jsonrpc_version():
    server = _make_server()
    payload = {"jsonrpc": "1.0", "id": 5, "method": "initialize", "params": {}}
    response = server.handle_request(payload)
    assert "error" in response
    assert response["error"]["code"] == -32600


# ---------------------------------------------------------------------------
# StdioMCPServer line processing (malformed JSON)
# ---------------------------------------------------------------------------


def test_malformed_json_line_does_not_crash():
    server = _make_server()
    response = server._process_line("{this is not json}")
    assert "error" in response
    assert response["error"]["code"] == -32700


def test_non_object_json_returns_error():
    server = _make_server()
    response = server._process_line('"just a string"')
    assert "error" in response


# ---------------------------------------------------------------------------
# StdioMCPServer start / stop
# ---------------------------------------------------------------------------


def test_start_stop_lifecycle():
    server = _make_server()
    assert not server._running
    server.start()
    assert server._running
    server.stop()
    assert not server._running


# ---------------------------------------------------------------------------
# MCPServerConfig validation
# ---------------------------------------------------------------------------


def test_config_invalid_transport():
    with pytest.raises(ValueError, match="transport must be one of"):
        MCPServerConfig(transport="websocket")


def test_config_invalid_port():
    with pytest.raises(ValueError, match="port must be in"):
        MCPServerConfig(port=0)


def test_config_invalid_timeout():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        MCPServerConfig(timeout_s=-1.0)


# ---------------------------------------------------------------------------
# run_loop end-to-end via StringIO
# ---------------------------------------------------------------------------


def test_run_loop_processes_request():
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}) + "\n"
    stdin = io.StringIO(request)
    stdout = io.StringIO()
    server = StdioMCPServer(stdin=stdin, stdout=stdout)
    server.run_loop()
    output = stdout.getvalue().strip()
    response = json.loads(output)
    assert response["jsonrpc"] == "2.0"
    assert "result" in response
