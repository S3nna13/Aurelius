"""Aurelius MCP server abstraction layer.

Provides transport-agnostic server classes for the Model Context Protocol,
including a stdio-based JSON-RPC implementation using only stdlib dependencies.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

import io
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_TRANSPORTS = frozenset({"stdio", "sse", "http"})


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server instance."""

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8080
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {sorted(_VALID_TRANSPORTS)!r}, "
                f"got {self.transport!r}"
            )
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not (0 < self.port < 65536):
            raise ValueError(f"port must be in 1..65535, got {self.port}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MCPServer(ABC):
    """Abstract base class for MCP server implementations."""

    @abstractmethod
    def start(self) -> None:
        """Start the server (bind ports, open streams, etc.)."""

    @abstractmethod
    def stop(self) -> None:
        """Gracefully stop the server and release resources."""

    @abstractmethod
    def handle_request(self, payload: dict) -> dict:
        """Process a single JSON-RPC request dict and return a response dict."""


# ---------------------------------------------------------------------------
# Stdio implementation
# ---------------------------------------------------------------------------

_JSONRPC_VERSION = "2.0"
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603


def _error_response(id_: object, code: int, message: str) -> dict:
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": id_,
        "error": {"code": code, "message": message},
    }


class StdioMCPServer(MCPServer):
    """MCP server that communicates via newline-delimited JSON-RPC on stdin/stdout.

    Uses only stdlib ``io`` — no external dependencies.
    The server runs a simple request/response loop: one JSON object per line in,
    one JSON object per line out.
    """

    def __init__(
        self,
        config: MCPServerConfig | None = None,
        *,
        stdin: io.TextIOBase | None = None,
        stdout: io.TextIOBase | None = None,
    ) -> None:
        self.config = config or MCPServerConfig(transport="stdio")
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._running = False

    # ------------------------------------------------------------------
    # MCPServer protocol
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Set the server into running state (does not block; call run_loop for that)."""
        self._running = True

    def stop(self) -> None:
        """Signal the server loop to terminate."""
        self._running = False

    def handle_request(self, payload: dict) -> dict:
        """Dispatch a JSON-RPC request dict; return a well-formed response dict.

        Unknown methods return a ``-32601`` error rather than raising.
        """
        req_id = payload.get("id")

        if payload.get("jsonrpc") != _JSONRPC_VERSION:
            return _error_response(
                req_id, _INVALID_REQUEST, "jsonrpc field must be '2.0'"
            )

        method = payload.get("method")
        if not isinstance(method, str) or not method:
            return _error_response(
                req_id, _INVALID_REQUEST, "missing or invalid 'method' field"
            )

        params = payload.get("params") or {}
        handler = self._METHOD_HANDLERS.get(method)
        if handler is None:
            return _error_response(
                req_id, _METHOD_NOT_FOUND, f"method not found: {method!r}"
            )

        try:
            result = handler(self, params)
            return {"jsonrpc": _JSONRPC_VERSION, "id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            return _error_response(req_id, _INTERNAL_ERROR, str(exc))

    # ------------------------------------------------------------------
    # Built-in method handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": "aurelius-stdio-mcp", "version": "1.0.0"},
        }

    def _handle_tools_list(self, params: dict) -> dict:
        return {"tools": []}

    def _handle_ping(self, params: dict) -> dict:
        return {}

    _METHOD_HANDLERS: ClassVar[dict] = {
        "initialize": _handle_initialize,
        "tools/list": _handle_tools_list,
        "ping": _handle_ping,
    }

    # ------------------------------------------------------------------
    # Line-oriented loop (for real stdio usage)
    # ------------------------------------------------------------------

    def run_loop(self) -> None:
        """Read newline-delimited JSON-RPC from stdin and write responses to stdout.

        Exits when stdin is exhausted or ``stop()`` is called.
        """
        self._running = True
        for line in self._stdin:
            if not self._running:
                break
            line = line.strip()
            if not line:
                continue
            response = self._process_line(line)
            self._stdout.write(json.dumps(response) + "\n")
            self._stdout.flush()

    def _process_line(self, line: str) -> dict:
        """Parse a single JSON line and return a JSON-RPC response dict."""
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return _error_response(None, _PARSE_ERROR, f"parse error: {exc}")
        if not isinstance(payload, dict):
            return _error_response(None, _INVALID_REQUEST, "request must be a JSON object")
        return self.handle_request(payload)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps transport names to MCPServer subclasses.
MCP_SERVER_REGISTRY: dict[str, type[MCPServer]] = {}


def register_mcp_server(name: str, cls: type[MCPServer]) -> None:
    """Register an MCPServer subclass under *name*.

    Raises ``TypeError`` if *cls* is not a subclass of ``MCPServer``.
    """
    if not (isinstance(cls, type) and issubclass(cls, MCPServer)):
        raise TypeError(f"{cls!r} must be a subclass of MCPServer")
    MCP_SERVER_REGISTRY[name] = cls


def get_mcp_server(name: str) -> type[MCPServer]:
    """Return the MCPServer class registered under *name*.

    Raises ``KeyError`` with a descriptive message if not found.
    """
    if name not in MCP_SERVER_REGISTRY:
        raise KeyError(
            f"No MCP server registered under {name!r}. "
            f"Available: {sorted(MCP_SERVER_REGISTRY)!r}"
        )
    return MCP_SERVER_REGISTRY[name]


def list_mcp_servers() -> list[str]:
    """Return a sorted list of all registered MCP server names."""
    return sorted(MCP_SERVER_REGISTRY)


# Pre-populate the registry.
register_mcp_server("stdio", StdioMCPServer)


__all__ = [
    "MCPServer",
    "MCPServerConfig",
    "MCP_SERVER_REGISTRY",
    "StdioMCPServer",
    "get_mcp_server",
    "list_mcp_servers",
    "register_mcp_server",
]
