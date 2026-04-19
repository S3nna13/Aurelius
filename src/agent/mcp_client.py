"""Model Context Protocol (MCP) client shim.

This module implements the client-side of Anthropic's open Model Context
Protocol (MCP) spec v2024-11-05 (see https://modelcontextprotocol.io/).
It is transport-agnostic: callers inject a ``call_fn(method, params)``
callable that performs the actual JSON-RPC round-trip over whichever
transport they prefer (stdio, SSE, HTTP). The client concerns itself
only with:

    * Building well-formed JSON-RPC 2.0 request dicts.
    * Incrementing the request id across calls for determinism.
    * Validating the shape of responses returned by ``call_fn``.
    * Translating payloads to/from small, ergonomic dataclasses.

No sockets, no pipes, no HTTP libraries, no ``mcp`` package — pure
stdlib (``json``, ``dataclasses``). No silent fallbacks: malformed or
error responses raise :class:`MCPProtocolError` loudly.

The ``call_fn`` contract is:

    call_fn(method: str, params: dict) -> dict

where the returned dict is the JSON-RPC response envelope, i.e. either
``{"jsonrpc": "2.0", "id": ..., "result": {...}}`` or
``{"jsonrpc": "2.0", "id": ..., "error": {"code": int, "message": str}}``.

The client increments and injects the ``id`` before calling ``call_fn``
so that transports that need it can echo it back; the returned ``id``
is not strictly validated (some transports, e.g. stdio-framed servers,
strip it) but ``result`` / ``error`` presence is.
"""

from __future__ import annotations

import json  # noqa: F401 — imported to satisfy "pure stdlib" contract & for callers
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPProtocolError(RuntimeError):
    """Raised when an MCP response is malformed, errored, or unexpected.

    This is the single error class surfaced by the client. It wraps
    both transport-layer failures (``call_fn`` raising) and protocol
    violations (missing ``result``, server ``error`` payloads, shape
    mismatches) so callers have one thing to catch.
    """


@dataclass(frozen=True)
class MCPToolSpec:
    """An MCP tool descriptor as returned by ``tools/list``."""

    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class MCPToolCallResult:
    """Result of an MCP ``tools/call``.

    ``content`` mirrors the MCP content-block list (each block is a
    dict with a ``type`` key and type-specific payload, e.g.
    ``{"type": "text", "text": "..."}``). ``is_error`` reflects the
    server-signalled ``isError`` flag; a ``True`` value does NOT cause
    the client to raise — tool errors are data, not exceptions — but
    transport / protocol errors still raise.
    """

    content: list[dict]
    is_error: bool
    tool_name: str


@dataclass(frozen=True)
class MCPResource:
    """An MCP resource descriptor as returned by ``resources/list``."""

    uri: str
    name: str
    mime_type: str


@dataclass(frozen=True)
class MCPPrompt:
    """An MCP prompt descriptor as returned by ``prompts/list``."""

    name: str
    description: str
    arguments: list[dict] = field(default_factory=list)


def _require(obj: Any, key: str, method: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise MCPProtocolError(
            f"MCP response for {method!r} missing required field {key!r}: {obj!r}"
        )
    return obj[key]


def _unwrap_result(method: str, response: Any) -> dict:
    """Validate a JSON-RPC envelope and return its ``result`` dict."""

    if not isinstance(response, dict):
        raise MCPProtocolError(
            f"MCP response for {method!r} is not a dict: {response!r}"
        )
    if "error" in response:
        err = response["error"] or {}
        code = err.get("code", "?") if isinstance(err, dict) else "?"
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise MCPProtocolError(
            f"MCP server returned error for {method!r} (code={code}): {msg}"
        )
    if "result" not in response:
        raise MCPProtocolError(
            f"MCP response for {method!r} missing 'result' and 'error': {response!r}"
        )
    result = response["result"]
    if not isinstance(result, dict):
        raise MCPProtocolError(
            f"MCP 'result' for {method!r} is not a dict: {result!r}"
        )
    return result


class MCPClient:
    """A transport-agnostic MCP client.

    Parameters
    ----------
    call_fn:
        Callable that performs one JSON-RPC round-trip given
        ``(method, params)`` and returns the response envelope.
    client_name / client_version:
        Identifying info sent during :meth:`initialize` handshake.

    Notes
    -----
    The client maintains a monotonically increasing request id
    (starting at 1) that is stamped into each request's ``params``
    under the ``_jsonrpc_id`` key for transports that want to forward
    it, and is also the visible value of :attr:`last_request_id`.
    """

    def __init__(
        self,
        call_fn: Callable[[str, dict], dict],
        client_name: str = "aurelius",
        client_version: str = "0.1.0",
    ) -> None:
        if not callable(call_fn):
            raise TypeError("call_fn must be callable")
        self._call_fn = call_fn
        self.client_name = client_name
        self.client_version = client_version
        self._next_id = 1
        self.last_request_id: int = 0

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------
    def _call(self, method: str, params: Optional[dict] = None) -> dict:
        rid = self._next_id
        self._next_id += 1
        self.last_request_id = rid
        payload = dict(params or {})
        payload["_jsonrpc_id"] = rid
        try:
            response = self._call_fn(method, payload)
        except MCPProtocolError:
            raise
        except Exception as exc:  # transport failure — wrap with context
            raise MCPProtocolError(
                f"MCP transport raised while calling {method!r}: {exc!r}"
            ) from exc
        return _unwrap_result(method, response)

    # ------------------------------------------------------------------
    # handshake
    # ------------------------------------------------------------------
    def initialize(self) -> dict:
        """Perform the MCP ``initialize`` handshake.

        Returns the server's raw ``result`` dict (typically
        ``{"protocolVersion": ..., "capabilities": {...}, "serverInfo": {...}}``).
        """

        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": self.client_name,
                "version": self.client_version,
            },
        }
        return self._call("initialize", params)

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    def list_tools(self) -> list[MCPToolSpec]:
        result = self._call("tools/list", {})
        raw = _require(result, "tools", "tools/list")
        if not isinstance(raw, list):
            raise MCPProtocolError(
                f"MCP tools/list 'tools' is not a list: {raw!r}"
            )
        out: list[MCPToolSpec] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise MCPProtocolError(
                    f"MCP tools/list entry is not a dict: {entry!r}"
                )
            name = _require(entry, "name", "tools/list")
            schema = _require(entry, "inputSchema", "tools/list")
            description = entry.get("description", "")
            out.append(
                MCPToolSpec(
                    name=str(name),
                    description=str(description),
                    input_schema=dict(schema),
                )
            )
        return out

    def call_tool(self, name: str, arguments: dict) -> MCPToolCallResult:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be a dict")
        result = self._call(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        content = _require(result, "content", "tools/call")
        if not isinstance(content, list):
            raise MCPProtocolError(
                f"MCP tools/call 'content' is not a list: {content!r}"
            )
        is_error = bool(result.get("isError", False))
        return MCPToolCallResult(
            content=[dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)}
                     for b in content],
            is_error=is_error,
            tool_name=name,
        )

    # ------------------------------------------------------------------
    # resources
    # ------------------------------------------------------------------
    def list_resources(self) -> list[MCPResource]:
        result = self._call("resources/list", {})
        raw = _require(result, "resources", "resources/list")
        if not isinstance(raw, list):
            raise MCPProtocolError(
                f"MCP resources/list 'resources' is not a list: {raw!r}"
            )
        out: list[MCPResource] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise MCPProtocolError(
                    f"MCP resources/list entry is not a dict: {entry!r}"
                )
            uri = _require(entry, "uri", "resources/list")
            name = entry.get("name", str(uri))
            mime = entry.get("mimeType", "application/octet-stream")
            out.append(
                MCPResource(uri=str(uri), name=str(name), mime_type=str(mime))
            )
        return out

    def read_resource(self, uri: str) -> str:
        if not isinstance(uri, str) or not uri:
            raise ValueError("uri must be a non-empty string")
        result = self._call("resources/read", {"uri": uri})
        contents = _require(result, "contents", "resources/read")
        if not isinstance(contents, list) or not contents:
            raise MCPProtocolError(
                f"MCP resources/read 'contents' must be a non-empty list: {contents!r}"
            )
        parts: list[str] = []
        for block in contents:
            if not isinstance(block, dict):
                raise MCPProtocolError(
                    f"MCP resources/read content block is not a dict: {block!r}"
                )
            if "text" in block:
                parts.append(str(block["text"]))
            elif "blob" in block:
                # base64 blob — preserve as-is; callers decode.
                parts.append(str(block["blob"]))
            else:
                raise MCPProtocolError(
                    f"MCP resources/read block missing 'text' and 'blob': {block!r}"
                )
        return "".join(parts)

    # ------------------------------------------------------------------
    # prompts
    # ------------------------------------------------------------------
    def list_prompts(self) -> list[MCPPrompt]:
        result = self._call("prompts/list", {})
        raw = _require(result, "prompts", "prompts/list")
        if not isinstance(raw, list):
            raise MCPProtocolError(
                f"MCP prompts/list 'prompts' is not a list: {raw!r}"
            )
        out: list[MCPPrompt] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise MCPProtocolError(
                    f"MCP prompts/list entry is not a dict: {entry!r}"
                )
            name = _require(entry, "name", "prompts/list")
            description = entry.get("description", "")
            arguments = entry.get("arguments", []) or []
            if not isinstance(arguments, list):
                raise MCPProtocolError(
                    f"MCP prompts/list 'arguments' is not a list: {arguments!r}"
                )
            out.append(
                MCPPrompt(
                    name=str(name),
                    description=str(description),
                    arguments=[dict(a) for a in arguments if isinstance(a, dict)],
                )
            )
        return out

    def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("prompt name must be a non-empty string")
        params: dict = {"name": name}
        if arguments is not None:
            if not isinstance(arguments, dict):
                raise TypeError("arguments must be a dict or None")
            params["arguments"] = arguments
        result = self._call("prompts/get", params)
        messages = _require(result, "messages", "prompts/get")
        if not isinstance(messages, list):
            raise MCPProtocolError(
                f"MCP prompts/get 'messages' is not a list: {messages!r}"
            )
        rendered: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise MCPProtocolError(
                    f"MCP prompts/get message is not a dict: {msg!r}"
                )
            content = msg.get("content")
            role = msg.get("role", "user")
            if isinstance(content, dict) and "text" in content:
                rendered.append(f"{role}: {content['text']}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        rendered.append(f"{role}: {block['text']}")
                    else:
                        raise MCPProtocolError(
                            f"MCP prompts/get unsupported content block: {block!r}"
                        )
            elif isinstance(content, str):
                rendered.append(f"{role}: {content}")
            else:
                raise MCPProtocolError(
                    f"MCP prompts/get message has unsupported content: {msg!r}"
                )
        return "\n".join(rendered)


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCPClient",
    "MCPPrompt",
    "MCPProtocolError",
    "MCPResource",
    "MCPToolCallResult",
    "MCPToolSpec",
]
