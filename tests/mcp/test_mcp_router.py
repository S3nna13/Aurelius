"""Tests for src/mcp/mcp_router.py — ≥28 test cases."""

from __future__ import annotations

import dataclasses

import pytest

from src.mcp.mcp_router import (
    MCP_ROUTER_REGISTRY,
    MCPRouter,
    RouteMatch,
    RoutePattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def echo_handler(payload: dict) -> dict:
    return {"ok": True, "payload": payload}


def make_router(*routes: tuple[str, str]) -> MCPRouter:
    """Build a router with (path, method) pairs all mapped to echo_handler."""
    router = MCPRouter()
    for path, method in routes:
        router.add_route(RoutePattern(path=path, method=method), echo_handler)
    return router


# ---------------------------------------------------------------------------
# RoutePattern frozen dataclass
# ---------------------------------------------------------------------------


class TestRoutePatternFrozen:
    def test_is_frozen(self):
        pat = RoutePattern(path="/foo")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            pat.path = "/bar"  # type: ignore[misc]

    def test_defaults(self):
        pat = RoutePattern(path="/x")
        assert pat.method == "*"
        assert pat.handler_name == ""

    def test_custom_fields(self):
        pat = RoutePattern(path="/p", method="POST", handler_name="h")
        assert pat.path == "/p"
        assert pat.method == "POST"
        assert pat.handler_name == "h"


# ---------------------------------------------------------------------------
# RouteMatch frozen dataclass
# ---------------------------------------------------------------------------


class TestRouteMatchFrozen:
    def test_is_frozen(self):
        pat = RoutePattern(path="/foo")
        rm = RouteMatch(pattern=pat, path_params={})
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            rm.path_params = {"x": "1"}  # type: ignore[misc]

    def test_holds_pattern_and_params(self):
        pat = RoutePattern(path="/a/{id}")
        rm = RouteMatch(pattern=pat, path_params={"id": "42"})
        assert rm.pattern is pat
        assert rm.path_params == {"id": "42"}


# ---------------------------------------------------------------------------
# add_route / match basics
# ---------------------------------------------------------------------------


class TestAddRouteAndMatch:
    def test_exact_match(self):
        router = make_router(("/ping", "*"))
        m = router.match("/ping")
        assert m is not None
        assert m.pattern.path == "/ping"

    def test_no_match_returns_none(self):
        router = make_router(("/ping", "*"))
        assert router.match("/pong") is None

    def test_multiple_routes_first_registered_wins_same_specificity(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/a"), lambda p: {"src": "first"})
        router.add_route(RoutePattern(path="/a"), lambda p: {"src": "second"})
        result = router.dispatch("/a", "*", {})
        assert result["src"] == "first"

    def test_add_multiple_distinct_routes(self):
        router = make_router(("/alpha", "*"), ("/beta", "*"))
        assert router.match("/alpha") is not None
        assert router.match("/beta") is not None

    def test_match_returns_route_match_type(self):
        router = make_router(("/x", "*"))
        m = router.match("/x")
        assert isinstance(m, RouteMatch)


# ---------------------------------------------------------------------------
# Parameterized path matching
# ---------------------------------------------------------------------------


class TestParamExtraction:
    def test_single_param(self):
        router = make_router(("/models/{model_id}", "*"))
        m = router.match("/models/gpt4")
        assert m is not None
        assert m.path_params == {"model_id": "gpt4"}

    def test_multi_param(self):
        router = make_router(("/a/{x}/b/{y}", "*"))
        m = router.match("/a/foo/b/bar")
        assert m is not None
        assert m.path_params == {"x": "foo", "y": "bar"}

    def test_predict_suffix(self):
        router = make_router(("/models/{model_id}/predict", "*"))
        m = router.match("/models/mymodel/predict")
        assert m is not None
        assert m.path_params == {"model_id": "mymodel"}

    def test_no_match_wrong_segment_count(self):
        router = make_router(("/models/{model_id}", "*"))
        assert router.match("/models/foo/extra") is None

    def test_param_empty_path_components_no_match(self):
        router = make_router(("/a/{b}/c", "*"))
        assert router.match("/a//c") is None


# ---------------------------------------------------------------------------
# Exact path takes priority over parameterized
# ---------------------------------------------------------------------------


class TestExactPriority:
    def test_exact_beats_param(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/models/{model_id}"), lambda p: {"src": "param"})
        router.add_route(RoutePattern(path="/models/special"), lambda p: {"src": "exact"})
        result = router.dispatch("/models/special", "*", {})
        assert result["src"] == "exact"

    def test_param_used_for_non_exact(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/models/{model_id}"), lambda p: {"src": "param"})
        router.add_route(RoutePattern(path="/models/special"), lambda p: {"src": "exact"})
        result = router.dispatch("/models/other", "*", {})
        assert result["src"] == "param"


# ---------------------------------------------------------------------------
# Method filtering
# ---------------------------------------------------------------------------


class TestMethodFilter:
    def test_method_specific_match(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/resource", method="GET"), echo_handler)
        assert router.match("/resource", "GET") is not None

    def test_method_mismatch_no_match(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/resource", method="GET"), echo_handler)
        assert router.match("/resource", "POST") is None

    def test_wildcard_method_matches_any(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/resource", method="*"), echo_handler)
        assert router.match("/resource", "DELETE") is not None

    def test_request_wildcard_matches_method_pattern(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/resource", method="GET"), echo_handler)
        assert router.match("/resource", "*") is not None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_dispatch_calls_handler(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/ping"), echo_handler)
        result = router.dispatch("/ping", "*", {"data": 1})
        assert result["ok"] is True

    def test_dispatch_injects_path_params(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/items/{item_id}"), echo_handler)
        result = router.dispatch("/items/99", "*", {})
        assert result["payload"]["item_id"] == "99"

    def test_dispatch_merges_payload_and_params(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/x/{id}"), echo_handler)
        result = router.dispatch("/x/7", "*", {"extra": "val"})
        assert result["payload"]["extra"] == "val"
        assert result["payload"]["id"] == "7"

    def test_dispatch_no_match_raises_key_error(self):
        router = MCPRouter()
        with pytest.raises(KeyError):
            router.dispatch("/nothing", "GET", {})

    def test_dispatch_method_filter_no_match_raises(self):
        router = MCPRouter()
        router.add_route(RoutePattern(path="/a", method="GET"), echo_handler)
        with pytest.raises(KeyError):
            router.dispatch("/a", "POST", {})


# ---------------------------------------------------------------------------
# list_routes
# ---------------------------------------------------------------------------


class TestListRoutes:
    def test_sorted_output(self):
        router = make_router(("/z", "*"), ("/a", "*"), ("/m", "*"))
        assert router.list_routes() == ["/a", "/m", "/z"]

    def test_empty_router(self):
        router = MCPRouter()
        assert router.list_routes() == []

    def test_single_route(self):
        router = make_router(("/only", "*"))
        assert router.list_routes() == ["/only"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in MCP_ROUTER_REGISTRY

    def test_registry_default_is_mcp_router(self):
        assert MCP_ROUTER_REGISTRY["default"] is MCPRouter

    def test_registry_instantiable(self):
        cls = MCP_ROUTER_REGISTRY["default"]
        router = cls()
        assert isinstance(router, MCPRouter)
