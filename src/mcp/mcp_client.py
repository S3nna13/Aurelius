"""Aurelius MCP protocol client (stub, no actual network).

Provides a transport-agnostic stub client for the Model Context Protocol,
with request/response logging and configurable retry behaviour.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from __future__ import annotations

if TYPE_CHECKING:
    from src.mcp.mcp_server import StdioMCPServer


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPRequest:
    """Represents a single outbound MCP protocol request.

    Attributes:
        method:     RPC method name (e.g. "tools/call").
        params:     Arbitrary key/value parameters for the method.
        request_id: Unique identifier; auto-assigned via uuid4 if empty.
    """

    method: str
    params: dict
    request_id: str = ""

    def __post_init__(self) -> None:
        # Assign a random ID when the caller leaves it blank.
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid.uuid4().hex[:8])


@dataclass(frozen=True)
class MCPResponse:
    """Represents a single inbound MCP protocol response.

    Attributes:
        request_id: Echoes the originating ``MCPRequest.request_id``.
        result:     Payload dict on success; ``None`` on error.
        error:      Human-readable error message; ``None`` on success.
    """

    request_id: str
    result: dict | None
    error: str | None = None


@dataclass(frozen=True)
class MCPClientConfig:
    """Configuration for an ``MCPClient`` instance.

    Attributes:
        server_name: Logical name of the target MCP server.
        timeout_ms:  Per-call timeout in milliseconds (default 5 000).
        max_retries: Maximum retry attempts for ``call_with_retry`` (default 3).
        transport:   Transport type ("local", "stdio", "sse", etc.). Default "local".
    """

    server_name: str
    timeout_ms: int = 5000
    max_retries: int = 3
    transport: str = "local"


# ---------------------------------------------------------------------------
# Default transport
# ---------------------------------------------------------------------------


def _default_transport(req: MCPRequest) -> MCPResponse:
    """Echo transport: returns the request params under the key 'echo'."""
    return MCPResponse(request_id=req.request_id, result={"echo": req.params})


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MCPClient:
    """Stub MCP protocol client.

    Parameters
    ----------
    config:
        Client configuration (server name, timeout, retry limit).
    transport_fn:
        Callable that accepts an ``MCPRequest`` and returns an
        ``MCPResponse``.  Defaults to :func:`_default_transport` which
        echoes the request params back to the caller.
    """

    def __init__(
        self,
        config: MCPClientConfig,
        transport_fn: Callable[[MCPRequest], MCPResponse] | None = None,
    ) -> None:
        self._config = config
        self._transport_fn: Callable[[MCPRequest], MCPResponse] = (
            transport_fn if transport_fn is not None else _default_transport
        )
        self._request_log: list[MCPRequest] = []
        self._response_log: list[MCPResponse] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def request_log(self) -> list[MCPRequest]:
        """Ordered list of every request dispatched by this client."""
        return list(self._request_log)

    @property
    def response_log(self) -> list[MCPResponse]:
        """Ordered list of every response received by this client."""
        return list(self._response_log)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _dispatch(self, method: str, params: dict) -> MCPResponse:
        """Build an MCPRequest, call the transport, and log both ends."""
        req = MCPRequest(method=method, params=params)
        self._request_log.append(req)
        resp = self._transport_fn(req)
        self._response_log.append(resp)
        return resp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, method: str, params: dict) -> MCPResponse:
        """Dispatch a single MCP call and return the response."""
        return self._dispatch(method, params)

    def call_with_retry(self, method: str, params: dict) -> MCPResponse:
        """Dispatch an MCP call, retrying up to ``config.max_retries`` times.

        Retries are triggered when ``MCPResponse.error`` is not ``None``.
        The last response is returned regardless of whether it succeeded.
        """
        resp: MCPResponse | None = None
        for _ in range(self._config.max_retries):
            resp = self._dispatch(method, params)
            if resp.error is None:
                return resp
        # Return whatever the last attempt produced.
        assert resp is not None  # max_retries >= 1 guaranteed by usage
        return resp


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps logical client names to ``MCPClient`` (sub)classes.
MCP_CLIENT_REGISTRY: dict[str, type[MCPClient]] = {"default": MCPClient}


class LocalMCPClient(MCPClient):
    """Local in-process MCP client backed by a StdioMCPServer.

    Wraps a running ``StdioMCPServer`` and exposes the standard
    ``MCPClient`` interface with the server's local transport.
    """

    def __init__(
        self,
        server: StdioMCPServer,
        config: MCPClientConfig,
    ) -> None:
        self._server = server
        self._config = config
        self._request_log: list[MCPRequest] = []
        self._response_log: list[MCPResponse] = []

    @property
    def request_log(self) -> list[MCPRequest]:
        return list(self._request_log)

    @property
    def response_log(self) -> list[MCPResponse]:
        return list(self._response_log)

    def list_tools(self) -> list[dict]:
        """Call the server's tools/list method and return the result list."""
        req = MCPRequest(method="tools/list", params={})
        self._request_log.append(req)
        resp = self._server.handle_request({"method": "tools/list"})
        err = resp.get("error") if isinstance(resp, dict) else None
        self._response_log.append(MCPResponse(request_id=req.request_id, result=resp, error=err))
        return resp.get("tools", []) if isinstance(resp, dict) else []

    def call(self, method: str, params: dict) -> MCPResponse:
        req = MCPRequest(method=method, params=params)
        self._request_log.append(req)
        resp = self._server.handle_request({"method": method, **params})
        err = resp.get("error") if isinstance(resp, dict) else None
        self._response_log.append(MCPResponse(request_id=req.request_id, result=resp, error=err))
        return self._response_log[-1]


__all__ = [
    "LocalMCPClient",
    "MCPClient",
    "MCPClientConfig",
    "MCPRequest",
    "MCPResponse",
    "MCP_CLIENT_REGISTRY",
]
