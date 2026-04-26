"""Tests for src.mcp.sse_mcp_server — SSE transport MCP server.

Covers SSEMCPServerConfig defaults, SSEMCPServer instantiation, request routing,
error handling, registry population, and start/stop lifecycle.
"""

from __future__ import annotations

import importlib
import random

from src.mcp.mcp_server import MCP_SERVER_REGISTRY
from src.mcp.sse_mcp_server import (
    SSE_SERVER_REGISTRY,
    SSEMCPServer,
    SSEMCPServerConfig,
)

# ---------------------------------------------------------------------------
# SSEMCPServerConfig defaults
# ---------------------------------------------------------------------------


class TestSSEMCPServerConfigDefaults:
    def test_default_host(self):
        cfg = SSEMCPServerConfig()
        assert cfg.host == "127.0.0.1"

    def test_default_port(self):
        cfg = SSEMCPServerConfig()
        assert cfg.port == 8765

    def test_default_path(self):
        cfg = SSEMCPServerConfig()
        assert cfg.path == "/events"

    def test_default_cors_origins(self):
        cfg = SSEMCPServerConfig()
        assert cfg.cors_origins == []

    def test_custom_values(self):
        cfg = SSEMCPServerConfig(
            host="0.0.0.0", port=9999, path="/mcp", cors_origins=["https://example.com"]  # noqa: S104
        )
        assert cfg.host == "0.0.0.0"  # noqa: S104
        assert cfg.port == 9999
        assert cfg.path == "/mcp"
        assert cfg.cors_origins == ["https://example.com"]


# ---------------------------------------------------------------------------
# SSEMCPServer instantiation (no network)
# ---------------------------------------------------------------------------


class TestSSEMCPServerInstantiation:
    def test_instantiate_with_default_config(self):
        server = SSEMCPServer()
        assert server is not None

    def test_instantiate_with_custom_config(self):
        cfg = SSEMCPServerConfig(port=19001)
        server = SSEMCPServer(config=cfg)
        assert server.config.port == 19001

    def test_no_tcp_server_before_start(self):
        server = SSEMCPServer()
        assert server._tcp_server is None

    def test_no_thread_before_start(self):
        server = SSEMCPServer()
        assert server._thread is None

    def test_empty_tool_handlers_on_init(self):
        server = SSEMCPServer()
        assert server._tool_handlers == {}


# ---------------------------------------------------------------------------
# handle_request routing
# ---------------------------------------------------------------------------


class TestHandleRequest:
    def setup_method(self, method) -> None:
        """Restore SSEMCPServer.handle_request to its original implementation
        before every test in this class.

        Other tests in the full suite may monkey-patch handle_request (or the
        enclosing module) to study error-path behaviour.  Without this reset,
        test_handler_exception_returns_error_dict can observe a patched version
        that leaks exception detail, making the assertion flaky.
        """
        mod = importlib.import_module("src.mcp.sse_mcp_server")
        # Re-bind the method from the freshly-imported module so that any
        # per-instance or per-class monkey-patch applied by earlier tests is
        # undone for this test's server instances.
        self._original_handle_request = mod.SSEMCPServer.handle_request

    def teardown_method(self, method) -> None:
        """Ensure handle_request is restored even if a test fails mid-patch."""
        mod = importlib.import_module("src.mcp.sse_mcp_server")
        mod.SSEMCPServer.handle_request = self._original_handle_request

    def _make_server(self) -> SSEMCPServer:
        return SSEMCPServer()

    def test_registered_handler_receives_payload(self):
        server = self._make_server()
        received: list[dict] = []

        def echo(payload: dict) -> dict:
            received.append(payload)
            return {"result": "ok"}

        server.register_tool_handler("echo", echo)
        response = server.handle_request({"tool": "echo", "arg": 42})
        assert response == {"result": "ok"}
        assert received[0]["arg"] == 42

    def test_handler_return_value_propagated(self):
        server = self._make_server()
        server.register_tool_handler("add", lambda p: {"sum": p["a"] + p["b"]})
        resp = server.handle_request({"tool": "add", "a": 3, "b": 4})
        assert resp == {"sum": 7}

    def test_unknown_tool_returns_error_dict(self):
        server = self._make_server()
        resp = server.handle_request({"tool": "nonexistent"})
        assert "error" in resp
        assert "nonexistent" in resp["error"]

    def test_missing_tool_key_returns_error_dict(self):
        server = self._make_server()
        resp = server.handle_request({})
        assert "error" in resp

    def test_handler_exception_returns_error_dict(self):
        server = self._make_server()

        def boom(p: dict) -> dict:
            raise ValueError("intentional boom")

        server.register_tool_handler("boom", boom)
        resp = server.handle_request({"tool": "boom"})
        assert "error" in resp
        assert "boom" not in resp["error"]  # exception detail must not leak (AUR-SEC-2026-0010)

    def test_method_key_also_routes(self):
        """handle_request should also accept 'method' as the routing key."""
        server = self._make_server()
        server.register_tool_handler("ping", lambda p: {"pong": True})
        resp = server.handle_request({"method": "ping"})
        assert resp == {"pong": True}


# ---------------------------------------------------------------------------
# register_tool_handler
# ---------------------------------------------------------------------------


class TestRegisterToolHandler:
    def test_stores_callable(self):
        server = SSEMCPServer()
        fn = lambda p: {}  # noqa: E731
        server.register_tool_handler("myfn", fn)
        assert server._tool_handlers["myfn"] is fn

    def test_overwrite_existing_handler(self):
        server = SSEMCPServer()
        server.register_tool_handler("t", lambda p: {"v": 1})
        server.register_tool_handler("t", lambda p: {"v": 2})
        assert server.handle_request({"tool": "t"}) == {"v": 2}

    def test_multiple_handlers_independent(self):
        server = SSEMCPServer()
        server.register_tool_handler("a", lambda p: {"who": "a"})
        server.register_tool_handler("b", lambda p: {"who": "b"})
        assert server.handle_request({"tool": "a"})["who"] == "a"
        assert server.handle_request({"tool": "b"})["who"] == "b"


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------


class TestRegistries:
    def test_sse_server_registry_contains_sse(self):
        assert "sse" in SSE_SERVER_REGISTRY

    def test_sse_server_registry_maps_to_class(self):
        assert SSE_SERVER_REGISTRY["sse"] is SSEMCPServer

    def test_mcp_server_registry_contains_sse(self):
        assert "sse" in MCP_SERVER_REGISTRY

    def test_mcp_server_registry_sse_is_sse_class(self):
        assert MCP_SERVER_REGISTRY["sse"] is SSEMCPServer


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestStartStop:
    def _random_port(self) -> int:
        return random.randint(19000, 19999)

    def test_start_stop_cycle_completes(self):
        port = self._random_port()
        cfg = SSEMCPServerConfig(host="127.0.0.1", port=port)
        server = SSEMCPServer(config=cfg)
        server.start()
        assert server._tcp_server is not None
        assert server._thread is not None
        server.stop()
        assert server._tcp_server is None
        assert server._thread is None

    def test_stop_before_start_is_noop(self):
        server = SSEMCPServer()
        server.stop()  # must not raise
        assert server._tcp_server is None

    def test_double_start_idempotent(self):
        port = self._random_port()
        cfg = SSEMCPServerConfig(host="127.0.0.1", port=port)
        server = SSEMCPServer(config=cfg)
        try:
            server.start()
            first_srv = server._tcp_server
            server.start()  # second call should be a no-op
            assert server._tcp_server is first_srv
        finally:
            server.stop()
