"""Aurelius MCP (Model Context Protocol) surface.

Exposes the server registry, client registry, and tool schema registry for
the Aurelius MCP integration layer.  All submodules are lazily imported to
avoid circular dependencies and keep startup cost near zero.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from importlib import import_module

# ---------------------------------------------------------------------------
# Lazy submodule loader
# ---------------------------------------------------------------------------

_MCP_SUBMODULES = ("mcp_server", "mcp_client", "tool_schema_registry")

__all__ = [
    "MCP_SERVER_REGISTRY",
    "MCP_CLIENT_REGISTRY",
    "MCP_TOOL_SCHEMA_REGISTRY",
    *_MCP_SUBMODULES,
]


def __getattr__(name: str):
    # Expose top-level registry aliases.
    if name == "MCP_SERVER_REGISTRY":
        from .mcp_server import MCP_SERVER_REGISTRY  # noqa: PLC0415
        globals()["MCP_SERVER_REGISTRY"] = MCP_SERVER_REGISTRY
        return MCP_SERVER_REGISTRY
    if name == "MCP_CLIENT_REGISTRY":
        from .mcp_client import MCP_CLIENT_REGISTRY  # noqa: PLC0415
        globals()["MCP_CLIENT_REGISTRY"] = MCP_CLIENT_REGISTRY
        return MCP_CLIENT_REGISTRY
    if name == "MCP_TOOL_SCHEMA_REGISTRY":
        from .tool_schema_registry import MCP_TOOL_SCHEMA_REGISTRY  # noqa: PLC0415
        globals()["MCP_TOOL_SCHEMA_REGISTRY"] = MCP_TOOL_SCHEMA_REGISTRY
        return MCP_TOOL_SCHEMA_REGISTRY
    if name in _MCP_SUBMODULES:
        module = import_module(f"src.mcp.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'src.mcp' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
