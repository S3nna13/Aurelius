"""Aurelius MCP — mcp_router.py

Routes MCP requests to appropriate handlers using path pattern matching.
Supports {param} style path parameters. Exact paths take priority over
parameterized patterns. All logic is pure Python stdlib; no external deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


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


class MCPRouter:
    """Routes MCP requests to registered handler callables."""

    def __init__(self) -> None:
        # list of (RoutePattern, handler, compiled_regex, param_names)
        self._routes: list[
            tuple[RoutePattern, Callable[[dict], dict], re.Pattern[str], list[str]]
        ] = []

    def add_route(
        self, pattern: RoutePattern, handler: Callable[[dict], dict]
    ) -> None:
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

    def dispatch(self, path: str, method: str, payload: dict) -> dict:
        """Find the handler for *path*/*method*, inject path_params into *payload*,
        and call handler(payload).  Raises KeyError if no route matches.
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

        assert handler is not None  # invariant: match() found it from self._routes

        merged_payload = {**payload, **route_match.path_params}
        return handler(merged_payload)

    def list_routes(self) -> list[str]:
        """Return sorted list of registered pattern paths."""
        return sorted(pat.path for pat, *_ in self._routes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_ROUTER_REGISTRY: dict[str, type[MCPRouter]] = {"default": MCPRouter}
