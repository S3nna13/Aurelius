"""Tests for src.mcp.mcp_client.

Coverage:
- LocalMCPClient.call_tool routes to registered server
- list_tools() returns a list
- Unknown tool name raises or returns error cleanly
- MCPClientConfig validation
- Registry helpers
"""

from __future__ import annotations

import pytest

from src.mcp.mcp_client import (
    MCP_CLIENT_REGISTRY,
    LocalMCPClient,
    MCPClientConfig,
    MCPClientError,
    get_mcp_client,
    list_mcp_clients,
    register_mcp_client,
)
from src.mcp.mcp_server import MCPServerConfig, StdioMCPServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_server():
    """A StdioMCPServer subclass that handles 'tools/call' with echo."""

    class EchoServer(StdioMCPServer):
        def _handle_tools_call(self, params: dict) -> dict:
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            if name == "echo":
                return {"output": arguments}
            raise KeyError(f"unknown tool: {name}")

        _METHOD_HANDLERS = {
            **StdioMCPServer._METHOD_HANDLERS,
            "tools/call": _handle_tools_call,
        }

    server = EchoServer(MCPServerConfig(transport="stdio"))
    server.start()
    return server


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_contains_local():
    assert "local" in MCP_CLIENT_REGISTRY
    assert MCP_CLIENT_REGISTRY["local"] is LocalMCPClient


def test_list_mcp_clients_non_empty():
    clients = list_mcp_clients()
    assert isinstance(clients, list)
    assert "local" in clients


def test_get_mcp_client_returns_class():
    cls = get_mcp_client("local")
    assert cls is LocalMCPClient


def test_get_mcp_client_unknown_raises():
    with pytest.raises(KeyError, match="No MCP client registered"):
        get_mcp_client("__nonexistent__")


def test_register_mcp_client_type_check():
    with pytest.raises(TypeError):
        register_mcp_client("bad", object)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LocalMCPClient tests
# ---------------------------------------------------------------------------


def test_list_tools_returns_list():
    server = StdioMCPServer(MCPServerConfig(transport="stdio"))
    server.start()
    client = LocalMCPClient(server)
    tools = client.list_tools()
    assert isinstance(tools, list)


def test_call_tool_routes_to_server():
    server = _make_echo_server()
    client = LocalMCPClient(server)
    result = client.call_tool("echo", {"message": "hello"})
    assert isinstance(result, dict)
    assert result.get("output") == {"message": "hello"}


def test_call_tool_unknown_raises_mcp_client_error():
    server = _make_echo_server()
    client = LocalMCPClient(server)
    with pytest.raises(MCPClientError):
        client.call_tool("unknown_tool_xyz", {})


def test_call_tool_empty_name_raises():
    server = StdioMCPServer(MCPServerConfig(transport="stdio"))
    client = LocalMCPClient(server)
    with pytest.raises(MCPClientError, match="non-empty string"):
        client.call_tool("", {})


# ---------------------------------------------------------------------------
# MCPClientConfig validation
# ---------------------------------------------------------------------------


def test_client_config_invalid_transport():
    with pytest.raises(ValueError, match="transport must be one of"):
        MCPClientConfig(transport="websocket")


def test_client_config_empty_server_name():
    with pytest.raises(ValueError, match="server_name must be a non-empty string"):
        MCPClientConfig(server_name="")


def test_client_config_defaults():
    cfg = MCPClientConfig()
    assert cfg.server_name == "default"
    assert cfg.transport == "local"
    assert cfg.endpoint == ""
