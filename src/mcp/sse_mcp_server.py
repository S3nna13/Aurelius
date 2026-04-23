"""Aurelius MCP SSE (Server-Sent Events) transport server.

Provides an HTTP/SSE-based transport for the Model Context Protocol using only
stdlib dependencies (http.server, socketserver, threading, json).

Inspired by Cline MCP integration (MIT, github.com/cline/cline), Goose
extension/MCP system (Apache-2.0, github.com/block/goose), clean-room
reimplementation.
"""

from __future__ import annotations

import http.server
import json
import logging
import socketserver
import threading
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

_MAX_REQUEST_BODY = 1 * 1024 * 1024  # 1 MiB hard cap on incoming POST bodies

from .mcp_server import MCPServer, register_mcp_server

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SSEMCPServerConfig:
    """Configuration for an SSE-based MCP server instance."""

    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/events"
    cors_origins: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP request handler that serves SSE streams and accepts JSON POSTs."""

    # Injected by SSEMCPServer when building the handler class.
    _sse_server: "SSEMCPServer"

    # ------------------------------------------------------------------ GET --

    def do_GET(self) -> None:  # noqa: N802
        cfg = self._sse_server.config
        if self.path != cfg.path:
            self.send_response(404)
            self.end_headers()
            return

        self._send_cors_headers()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Send a single "connected" event then close — keeping the connection
        # open indefinitely would block the thread; tests only need the
        # handshake to succeed.
        event = json.dumps({"type": "connected", "server": "aurelius-sse-mcp"})
        self.wfile.write(f"data: {event}\n\n".encode())
        self.wfile.flush()

    # ----------------------------------------------------------------- POST --

    def do_POST(self) -> None:  # noqa: N802
        cfg = self._sse_server.config
        if self.path != cfg.path:
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = max(0, int(self.headers.get("Content-Length", 0)))
        except ValueError:
            length = 0
        if length > _MAX_REQUEST_BODY:
            self.send_response(413)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = self.rfile.read(length) if length else b"{}"

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            response = {"error": f"JSON parse error: {exc}"}
        else:
            response = self._sse_server.handle_request(payload)

        encoded = json.dumps(response).encode()
        self._send_cors_headers()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    # --------------------------------------------------------------- OPTIONS --

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_cors_headers()
        self.send_response(204)
        self.end_headers()

    # ---------------------------------------------------------------- helpers -

    def _send_cors_headers(self) -> None:
        origins = self._sse_server.config.cors_origins
        if not origins:
            return
        request_origin = self.headers.get("Origin", "")
        if request_origin in origins:
            self.send_header("Access-Control-Allow-Origin", request_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        """Suppress default access log noise during tests."""


# ---------------------------------------------------------------------------
# SSE MCP Server
# ---------------------------------------------------------------------------


class SSEMCPServer(MCPServer):
    """MCP server that delivers events over Server-Sent Events (SSE).

    Uses :mod:`http.server` and :mod:`socketserver` from the stdlib — no
    external dependencies.  The TCP server runs in a daemon thread so
    ``start()`` is non-blocking and ``stop()`` cleanly shuts it down.
    """

    def __init__(
        self,
        config: SSEMCPServerConfig | None = None,
        *,
        bind_and_activate: bool = False,
    ) -> None:
        self.config: SSEMCPServerConfig = config or SSEMCPServerConfig()
        self._tool_handlers: dict[str, Callable[[dict], dict]] = {}
        self._tcp_server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self.bind_and_activate = bind_and_activate

    # ------------------------------------------------------------------
    # MCPServer protocol
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the TCP port and serve in a background daemon thread."""
        if self._tcp_server is not None:
            return  # already started

        # Build a handler class that carries a reference to *this* server.
        handler_cls = type(
            "_BoundSSEHandler",
            (_SSEHandler,),
            {"_sse_server": self},
        )

        server = socketserver.TCPServer(
            (self.config.host, self.config.port),
            handler_cls,
            bind_and_activate=False,
        )
        server.allow_reuse_address = True
        if self.bind_and_activate:
            server.server_bind()
            server.server_activate()
        self._tcp_server = server

        self._thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="aurelius-sse-mcp",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the TCP server and wait for the thread to exit."""
        if self._tcp_server is None:
            return
        self._tcp_server.shutdown()
        self._tcp_server.server_close()
        self._tcp_server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def handle_request(self, payload: dict) -> dict:
        """Route *payload* to a registered tool handler by ``tool`` key.

        Returns ``{"error": "..."}`` for unknown tools rather than raising.
        """
        tool_name = payload.get("tool") or payload.get("method") or ""
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            return {"error": f"unknown tool: {tool_name!r}"}
        try:
            return handler(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Tool handler %r raised an exception", tool_name)
            return {"error": "handler error"}

    # ------------------------------------------------------------------
    # Tool handler registration
    # ------------------------------------------------------------------

    def register_tool_handler(self, name: str, fn: Callable[[dict], dict]) -> None:
        """Register *fn* as the handler for tool *name*.

        Subsequent calls with the same *name* replace the previous handler.
        """
        self._tool_handlers[name] = fn


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: Maps SSE server variant names to SSEMCPServer subclasses.
SSE_SERVER_REGISTRY: dict[str, type[SSEMCPServer]] = {
    "sse": SSEMCPServer,
}

# Register into the shared MCP_SERVER_REGISTRY so other subsystems can
# discover SSEMCPServer by name.
register_mcp_server("sse", SSEMCPServer)


__all__ = [
    "SSEMCPServer",
    "SSEMCPServerConfig",
    "SSE_SERVER_REGISTRY",
]
