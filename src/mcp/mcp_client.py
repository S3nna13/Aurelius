"""Aurelius MCP client abstraction layer.

Provides transport-agnostic client classes for the Model Context Protocol,
including an in-process ``LocalMCPClient`` that routes directly to a registered
MCPServer instance with no network or socket dependencies.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcp_server import MCPServer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_TRANSPORTS = frozenset({"stdio", "sse", "http", "local"})


@dataclass
class MCPClientConfig:
    """Configuration for an MCP client instance."""

    server_name: str = "default"
    transport: str = "local"
    endpoint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.server_name, str) or not self.server_name:
            raise ValueError("server_name must be a non-empty string")
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {sorted(_VALID_TRANSPORTS)!r}, "
                f"got {self.transport!r}"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MCPClient(ABC):
    """Abstract base class for MCP client implementations."""

    @abstractmethod
    def call_tool(self, name: str, args: dict) -> dict:
        """Invoke the named tool with *args* and return the result dict.

        Raises ``MCPClientError`` for unrecoverable failures.
        """

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """Return a list of tool descriptors available on the connected server."""


class MCPClientError(Exception):
    """Raised for unrecoverable MCP client-side failures."""


# ---------------------------------------------------------------------------
# Local (in-process) implementation
# ---------------------------------------------------------------------------


class LocalMCPClient(MCPClient):
    """In-process MCP client that routes directly to an ``MCPServer`` instance.

    No network, no sockets — suitable for unit tests and tightly-coupled
    in-process usage.
    """

    def __init__(self, server: "MCPServer", config: MCPClientConfig | None = None) -> None:
        self._server = server
        self.config = config or MCPClientConfig(transport="local")

    def call_tool(self, name: str, args: dict) -> dict:
        """Route a ``tools/call`` JSON-RPC request to the backing server.

        Raises ``MCPClientError`` if the server returns a JSON-RPC error or
        if the tool name is not found.
        """
        if not isinstance(name, str) or not name:
            raise MCPClientError("tool name must be a non-empty string")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        response = self._server.handle_request(payload)
        if "error" in response:
            error = response["error"]
            raise MCPClientError(
                f"MCP server error [{error.get('code')}]: {error.get('message')}"
            )
        return response.get("result", {})

    def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by the backing server."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        response = self._server.handle_request(payload)
        if "error" in response:
            error = response["error"]
            raise MCPClientError(
                f"MCP server error [{error.get('code')}]: {error.get('message')}"
            )
        result = response.get("result", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise MCPClientError(
                f"Expected 'tools' to be a list, got {type(tools).__name__}"
            )
        return tools


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps transport names to MCPClient subclasses.
MCP_CLIENT_REGISTRY: dict[str, type[MCPClient]] = {}


def register_mcp_client(name: str, cls: type[MCPClient]) -> None:
    """Register an MCPClient subclass under *name*.

    Raises ``TypeError`` if *cls* is not a subclass of ``MCPClient``.
    """
    if not (isinstance(cls, type) and issubclass(cls, MCPClient)):
        raise TypeError(f"{cls!r} must be a subclass of MCPClient")
    MCP_CLIENT_REGISTRY[name] = cls


def get_mcp_client(name: str) -> type[MCPClient]:
    """Return the MCPClient class registered under *name*.

    Raises ``KeyError`` with a descriptive message if not found.
    """
    if name not in MCP_CLIENT_REGISTRY:
        raise KeyError(
            f"No MCP client registered under {name!r}. "
            f"Available: {sorted(MCP_CLIENT_REGISTRY)!r}"
        )
    return MCP_CLIENT_REGISTRY[name]


def list_mcp_clients() -> list[str]:
    """Return a sorted list of all registered MCP client names."""
    return sorted(MCP_CLIENT_REGISTRY)


# Pre-populate the registry.
register_mcp_client("local", LocalMCPClient)


__all__ = [
    "LocalMCPClient",
    "MCPClient",
    "MCPClientConfig",
    "MCPClientError",
    "MCP_CLIENT_REGISTRY",
    "get_mcp_client",
    "list_mcp_clients",
    "register_mcp_client",
]
