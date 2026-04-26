"""Unit tests for ``src.agent.mcp_client``.

All tests use a fake ``call_fn`` that returns canned JSON-RPC
responses; there is no real transport in play.
"""

from __future__ import annotations

import pytest

from src.agent.mcp_client import (
    MCP_PROTOCOL_VERSION,
    MCPClient,
    MCPPrompt,
    MCPProtocolError,
    MCPResource,
    MCPToolCallResult,
    MCPToolSpec,
)


def _ok(result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def _err(code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": message}}


class FakeServer:
    """Records calls and serves canned responses by method."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method: str, params: dict) -> dict:
        self.calls.append((method, dict(params)))
        if method not in self.responses:
            return _err(-32601, f"method not found: {method}")
        resp = self.responses[method]
        if callable(resp):
            return resp(params)
        return resp


# ---------------------------------------------------------------------------
# handshake
# ---------------------------------------------------------------------------
def test_initialize_returns_handshake_dict():
    server = FakeServer(
        {
            "initialize": _ok(
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stub", "version": "0.0.1"},
                }
            )
        }
    )
    client = MCPClient(server)
    info = client.initialize()
    assert info["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert info["serverInfo"]["name"] == "stub"


def test_initialize_sent_with_client_metadata():
    server = FakeServer({"initialize": _ok({"protocolVersion": MCP_PROTOCOL_VERSION})})
    client = MCPClient(server, client_name="aurelius", client_version="9.9.9")
    client.initialize()
    method, params = server.calls[0]
    assert method == "initialize"
    assert params["clientInfo"] == {"name": "aurelius", "version": "9.9.9"}
    assert params["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "capabilities" in params


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def test_list_tools_parses_into_dataclasses():
    server = FakeServer(
        {
            "tools/list": _ok(
                {
                    "tools": [
                        {
                            "name": "add",
                            "description": "add two ints",
                            "inputSchema": {"type": "object"},
                        },
                        {
                            "name": "echo",
                            "description": "",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"s": {"type": "string"}},
                            },
                        },
                    ]
                }
            )
        }
    )
    client = MCPClient(server)
    tools = client.list_tools()
    assert len(tools) == 2
    assert all(isinstance(t, MCPToolSpec) for t in tools)
    assert tools[0].name == "add"
    assert tools[0].description == "add two ints"
    assert tools[1].input_schema["properties"]["s"]["type"] == "string"


def test_call_tool_returns_result():
    server = FakeServer(
        {
            "tools/call": _ok(
                {
                    "content": [{"type": "text", "text": "42"}],
                    "isError": False,
                }
            )
        }
    )
    client = MCPClient(server)
    out = client.call_tool("add", {"a": 1, "b": 41})
    assert isinstance(out, MCPToolCallResult)
    assert out.tool_name == "add"
    assert out.is_error is False
    assert out.content == [{"type": "text", "text": "42"}]
    _, params = server.calls[0]
    assert params["name"] == "add"
    assert params["arguments"] == {"a": 1, "b": 41}


def test_call_tool_marks_is_error():
    server = FakeServer(
        {
            "tools/call": _ok(
                {
                    "content": [{"type": "text", "text": "division by zero"}],
                    "isError": True,
                }
            )
        }
    )
    client = MCPClient(server)
    out = client.call_tool("divide", {"a": 1, "b": 0})
    assert out.is_error is True
    assert "zero" in out.content[0]["text"]


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------
def test_list_resources_returns_dataclasses():
    server = FakeServer(
        {
            "resources/list": _ok(
                {
                    "resources": [
                        {"uri": "file:///a.txt", "name": "a", "mimeType": "text/plain"},
                        {"uri": "file:///b.bin"},
                    ]
                }
            )
        }
    )
    client = MCPClient(server)
    resources = client.list_resources()
    assert len(resources) == 2
    assert all(isinstance(r, MCPResource) for r in resources)
    assert resources[0].uri == "file:///a.txt"
    assert resources[0].mime_type == "text/plain"
    # defaults fill in for missing optional fields
    assert resources[1].name == "file:///b.bin"
    assert resources[1].mime_type == "application/octet-stream"


def test_read_resource_returns_text():
    server = FakeServer(
        {
            "resources/read": _ok(
                {
                    "contents": [
                        {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hello "},
                        {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "world"},
                    ]
                }
            )
        }
    )
    client = MCPClient(server)
    assert client.read_resource("file:///a.txt") == "hello world"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def test_list_prompts_returns_dataclasses():
    server = FakeServer(
        {
            "prompts/list": _ok(
                {
                    "prompts": [
                        {
                            "name": "greet",
                            "description": "say hi",
                            "arguments": [{"name": "who", "required": True}],
                        },
                        {"name": "bare"},
                    ]
                }
            )
        }
    )
    client = MCPClient(server)
    prompts = client.list_prompts()
    assert len(prompts) == 2
    assert all(isinstance(p, MCPPrompt) for p in prompts)
    assert prompts[0].name == "greet"
    assert prompts[0].arguments == [{"name": "who", "required": True}]
    assert prompts[1].arguments == []


def test_get_prompt_renders_messages():
    server = FakeServer(
        {
            "prompts/get": _ok(
                {
                    "description": "greeting",
                    "messages": [
                        {"role": "system", "content": {"type": "text", "text": "Be nice."}},
                        {"role": "user", "content": [{"type": "text", "text": "Hello, Ada!"}]},
                    ],
                }
            )
        }
    )
    client = MCPClient(server)
    rendered = client.get_prompt("greet", {"who": "Ada"})
    assert "system: Be nice." in rendered
    assert "user: Hello, Ada!" in rendered
    _, params = server.calls[0]
    assert params["name"] == "greet"
    assert params["arguments"] == {"who": "Ada"}


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------
def test_call_fn_raising_propagates_with_context():
    def boom(method, params):
        raise OSError("pipe closed")

    client = MCPClient(boom)
    with pytest.raises(MCPProtocolError) as exc:
        client.initialize()
    assert "initialize" in str(exc.value)
    assert "pipe closed" in str(exc.value)


def test_malformed_response_missing_fields_raises():
    # tools/list response lacks the required 'tools' key
    server = FakeServer({"tools/list": _ok({"oops": []})})
    client = MCPClient(server)
    with pytest.raises(MCPProtocolError):
        client.list_tools()


def test_server_error_raises():
    server = FakeServer({"tools/call": _err(-32601, "tool not found")})
    client = MCPClient(server)
    with pytest.raises(MCPProtocolError) as exc:
        client.call_tool("missing", {})
    assert "tool not found" in str(exc.value)


# ---------------------------------------------------------------------------
# determinism & id increment
# ---------------------------------------------------------------------------
def test_deterministic_identical_calls_identical_results():
    server = FakeServer(
        {"tools/list": _ok({"tools": [{"name": "t", "description": "", "inputSchema": {}}]})}
    )
    client = MCPClient(server)
    a = client.list_tools()
    b = client.list_tools()
    assert a == b


def test_request_id_increments_across_calls():
    server = FakeServer(
        {
            "initialize": _ok({"protocolVersion": MCP_PROTOCOL_VERSION}),
            "tools/list": _ok({"tools": []}),
            "resources/list": _ok({"resources": []}),
        }
    )
    client = MCPClient(server)
    client.initialize()
    assert client.last_request_id == 1
    client.list_tools()
    assert client.last_request_id == 2
    client.list_resources()
    assert client.last_request_id == 3
    # id must appear in the outgoing payloads as _jsonrpc_id
    ids = [p["_jsonrpc_id"] for _, p in server.calls]
    assert ids == [1, 2, 3]
