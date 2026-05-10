"""Aurelius MCP — mcp_router.py

Routes MCP requests to appropriate handlers using path pattern matching.
Supports {param} style path parameters. Exact paths take priority over
parameterized patterns. All logic is pure Python stdlib; no external deps.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mcp.mcp_gateway import MCPGateway

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutePattern:
    path: str
    method: str = "*"
    handler_name: str = ""


@dataclass(frozen=True)
class RouteMatch:
    pattern: RoutePattern
    path_params: dict[str, str]


# ---------------------------------------------------------------------------
# MCPRouter
# ---------------------------------------------------------------------------


def _pattern_to_regex(path: str) -> tuple[re.Pattern[str], list[str]]:
    """Convert a path like '/models/{model_id}/predict' to a regex and param names."""
    param_names: list[str] = []
    parts = re.split(r"(\{[^}]+\})", path)
    regex_parts: list[str] = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            name = part[1:-1]
            param_names.append(name)
            regex_parts.append(r"([^/]+)")
        else:
            regex_parts.append(re.escape(part))
    pattern_str = "^" + "".join(regex_parts) + "$"
    return re.compile(pattern_str), param_names


def _is_parameterized(path: str) -> bool:
    return "{" in path


def _extract_tool_name(path: str, payload: dict, path_params: dict) -> str:
    """Best-effort extraction of the tool name from route context."""
    # 1. Explicit payload keys used by MCP conventions
    for key in ("tool", "tool_name", "name"):
        if key in payload:
            return str(payload[key])
    # 2. Path parameter named 'tool_name'
    if "tool_name" in path_params:
        return str(path_params["tool_name"])
    # 3. Path parameter named 'tool'
    if "tool" in path_params:
        return str(path_params["tool"])
    # 4. Fall back to the route path itself
    return path


class MCPRouter:
    """Routes MCP requests to registered handler callables.

    When a :class:`~src.mcp.mcp_gateway.MCPGateway` is attached, every
    dispatch is transparently proxied through the gateway's security
    layers.
    """

    def __init__(self, gateway: MCPGateway | None = None) -> None:
        # list of (RoutePattern, handler, compiled_regex, param_names)
        self._routes: list[
            tuple[RoutePattern, Callable[[dict], dict], re.Pattern[str], list[str]]
        ] = []
        self._gateway: MCPGateway | None = gateway

    def add_route(self, pattern: RoutePattern, handler: Callable[[dict], dict]) -> None:
        """Register *handler* for the given *pattern*."""
        compiled, param_names = _pattern_to_regex(pattern.path)
        self._routes.append((pattern, handler, compiled, param_names))

    def match(self, path: str, method: str = "*") -> RouteMatch | None:
        """Return the first RouteMatch for *path*/*method*.

        Exact (non-parameterized) patterns are checked before parameterized ones.
        Within each tier, registration order is preserved.
        """
        exact: list[tuple[RoutePattern, re.Pattern[str], list[str]]] = []
        parameterized: list[tuple[RoutePattern, re.Pattern[str], list[str]]] = []

        for pat, _handler, compiled, param_names in self._routes:
            if _is_parameterized(pat.path):
                parameterized.append((pat, compiled, param_names))
            else:
                exact.append((pat, compiled, param_names))

        for candidates in (exact, parameterized):
            for pat, compiled, param_names in candidates:
                # Method filter: "*" on pattern or request matches any.
                if pat.method != "*" and method != "*" and pat.method != method:
                    continue
                m = compiled.match(path)
                if m:
                    params = dict(zip(param_names, m.groups()))
                    return RouteMatch(pattern=pat, path_params=params)
        return None

    def dispatch(
        self,
        path: str,
        method: str,
        payload: dict,
        caller_id: str = "",
        caller_role: str = "default",
    ) -> dict:
        """Find the handler for *path*/*method*, inject path_params into *payload*,
        and call handler(payload).  Raises KeyError if no route matches.

        When a gateway is configured and enabled, the call is proxied through
        the gateway's security checks (allowlist, rate limits, egress filter,
        etc.) before the handler is invoked.
        """
        route_match = self.match(path, method)
        if route_match is None:
            raise KeyError(f"No route matched: {method} {path}")

        # Locate the handler for this pattern
        handler: Callable[[dict], dict] | None = None
        for pat, h, _compiled, _names in self._routes:
            if pat is route_match.pattern:
                handler = h
                break

        assert handler is not None  # invariant: match() found it from self._routes  # noqa: S101

        merged_payload = {**payload}
        for k, v in route_match.path_params.items():
            if k not in ("method", "tool_name", "name", "tool"):
                merged_payload[k] = v

        if self._gateway is not None and getattr(
            getattr(self._gateway, "config", None), "enabled", False
        ):
            tool_name = _extract_tool_name(path, payload, route_match.path_params)
            return self._gateway.intercept(
                handler=handler,
                tool_name=tool_name,
                caller_id=caller_id or "anonymous",
                caller_role=caller_role,
                params=merged_payload,
            )

        return handler(merged_payload)

    def list_routes(self) -> list[str]:
        """Return sorted list of registered pattern paths."""
        return sorted(pat.path for pat, *_ in self._routes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_ROUTER_REGISTRY: dict[str, type[MCPRouter]] = {"default": MCPRouter}
