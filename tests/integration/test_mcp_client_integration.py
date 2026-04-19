"""End-to-end integration test for ``src.agent.mcp_client``.

Uses a stub in-process JSON-RPC server object as the transport. No
sockets, pipes, or external packages. Also verifies that the MCP
client symbols are exposed via ``src.agent`` and that prior agent
surface entries remain intact.
"""

from __future__ import annotations

import pytest

import src.agent as agent_surface
from src.agent import (
    MCPClient,
    MCPPrompt,
    MCPProtocolError,
    MCPResource,
    MCPToolCallResult,
    MCPToolSpec,
)
from src.agent.mcp_client import MCP_PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# stub server: a toy MCP server exposing a calculator tool, a readme
# resource, and a greet prompt. JSON-RPC over Python call.
# ---------------------------------------------------------------------------
class StubMCPServer:
    def __init__(self) -> None:
        self.initialized = False
        self._next_id_seen: list[int] = []

    def __call__(self, method: str, params: dict) -> dict:
        rid = params.get("_jsonrpc_id", 0)
        self._next_id_seen.append(rid)
        try:
            result = self._dispatch(method, params)
        except KeyError as exc:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"method not found: {exc}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            self.initialized = True
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "stub-mcp", "version": "1.0"},
            }
        if method == "tools/list":
            return {"tools": [{
                "name": "add",
                "description": "Add two integers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"},
                                   "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            }]}
        if method == "tools/call":
            name = params["name"]
            args = params["arguments"]
            if name == "add":
                total = int(args["a"]) + int(args["b"])
                return {"content": [{"type": "text", "text": str(total)}],
                        "isError": False}
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}],
                    "isError": True}
        if method == "resources/list":
            return {"resources": [{
                "uri": "mem:///README.md",
                "name": "README",
                "mimeType": "text/markdown",
            }]}
        if method == "resources/read":
            uri = params["uri"]
            if uri == "mem:///README.md":
                return {"contents": [{
                    "uri": uri, "mimeType": "text/markdown",
                    "text": "# stub server\n",
                }]}
            raise KeyError(uri)
        if method == "prompts/list":
            return {"prompts": [{
                "name": "greet",
                "description": "Greet a person.",
                "arguments": [{"name": "who", "required": True}],
            }]}
        if method == "prompts/get":
            who = (params.get("arguments") or {}).get("who", "world")
            return {"description": "greeting",
                    "messages": [{"role": "user",
                                  "content": {"type": "text",
                                              "text": f"Hello, {who}!"}}]}
        raise KeyError(method)


# ---------------------------------------------------------------------------
# surface integrity
# ---------------------------------------------------------------------------
def test_mcp_client_exposed_via_agent_surface():
    assert agent_surface.MCPClient is MCPClient
    for sym in ("MCPClient", "MCPToolSpec", "MCPToolCallResult",
                "MCPResource", "MCPPrompt", "MCPProtocolError"):
        assert sym in agent_surface.__all__


def test_prior_agent_surface_entries_intact():
    for prior in ("ReActLoop", "ToolRegistryDispatcher", "BeamPlanner",
                  "RecoveringDispatcher", "UnifiedDiffGenerator",
                  "ShellCommandPlanner", "CodeExecutionSandbox",
                  "CodeTestRunner", "TaskDecomposer", "parse_json", "parse_xml"):
        assert prior in agent_surface.__all__
        assert hasattr(agent_surface, prior)


# ---------------------------------------------------------------------------
# end-to-end against the stub server
# ---------------------------------------------------------------------------
def test_end_to_end_full_session():
    server = StubMCPServer()
    client = MCPClient(server, client_name="aurelius-it", client_version="0.0.1")

    handshake = client.initialize()
    assert server.initialized is True
    assert handshake["serverInfo"]["name"] == "stub-mcp"

    tools = client.list_tools()
    assert len(tools) == 1 and isinstance(tools[0], MCPToolSpec)
    assert tools[0].name == "add"

    result = client.call_tool("add", {"a": 2, "b": 40})
    assert isinstance(result, MCPToolCallResult)
    assert result.is_error is False
    assert result.content[0]["text"] == "42"

    resources = client.list_resources()
    assert len(resources) == 1 and isinstance(resources[0], MCPResource)
    assert resources[0].mime_type == "text/markdown"

    body = client.read_resource("mem:///README.md")
    assert body.startswith("# stub server")

    prompts = client.list_prompts()
    assert len(prompts) == 1 and isinstance(prompts[0], MCPPrompt)

    rendered = client.get_prompt("greet", {"who": "Ada"})
    assert "Hello, Ada!" in rendered

    # id monotonic across the full session
    assert server._next_id_seen == sorted(server._next_id_seen)
    assert server._next_id_seen[0] == 1
    assert server._next_id_seen[-1] == client.last_request_id


def test_end_to_end_server_error_surfaces():
    server = StubMCPServer()
    client = MCPClient(server)
    with pytest.raises(MCPProtocolError) as exc:
        client.read_resource("mem:///does-not-exist")
    assert "method not found" in str(exc.value) or "does-not-exist" in str(exc.value)


def test_end_to_end_tool_error_is_data_not_exception():
    server = StubMCPServer()
    client = MCPClient(server)
    out = client.call_tool("unknown_tool", {})
    assert out.is_error is True
    assert "unknown tool" in out.content[0]["text"]
