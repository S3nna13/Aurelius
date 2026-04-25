"""Root conftest.py — test isolation fixtures for Aurelius.

Provides autouse function-scoped fixtures that snapshot module-level singleton
registries before each test and restore them afterwards.  This prevents
state-pollution across tests when the full suite is run, regardless of
execution order.

Registries protected:
  - API_SHAPE_REGISTRY      (src.serving.openai_api_validator)
  - TOOL_REGISTRY           (src.tools.tool_registry)
  - MCP_SERVER_REGISTRY     (src.mcp.mcp_server)
  - MCP_TOOL_SCHEMA_REGISTRY (src.mcp.tool_schema_registry)
  - SSE_SERVER_REGISTRY     (src.mcp.sse_mcp_server)
  - SAFE_EXTRACTOR_REGISTRY (src.security.safe_archive)
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

import pytest
import torch


@pytest.fixture
def x_8() -> torch.Tensor:
    """8-node graph node features (8 x 64)."""
    return torch.randn(8, 64)


@pytest.fixture
def edge_index_8() -> torch.Tensor:
    """8-edge graph edge index (2 x 8)."""
    return torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]],
        dtype=torch.long,
    )


def _snapshot(obj: Any) -> Any:
    """Return a shallow copy of a dict-like registry."""
    return dict(obj)


def _restore(obj: Any, snapshot: dict) -> None:
    """Clear *obj* and repopulate from *snapshot*."""
    obj.clear()
    obj.update(snapshot)


# ---------------------------------------------------------------------------
# API_SHAPE_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_api_shape_registry():
    """Snapshot/restore API_SHAPE_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.serving.openai_api_validator")
        registry = mod.API_SHAPE_REGISTRY
    except Exception:
        yield
        return

    snap = _snapshot(registry)
    yield
    _restore(registry, snap)


# ---------------------------------------------------------------------------
# TOOL_REGISTRY  (ToolRegistry._specs and ._handlers are internal dicts)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_tool_registry():
    """Snapshot/restore TOOL_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.tools.tool_registry")
        registry = mod.TOOL_REGISTRY
    except Exception:
        yield
        return

    snap_specs = _snapshot(registry._specs)
    snap_handlers = _snapshot(registry._handlers)
    yield
    _restore(registry._specs, snap_specs)
    _restore(registry._handlers, snap_handlers)


# ---------------------------------------------------------------------------
# MCP_SERVER_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_mcp_server_registry():
    """Snapshot/restore MCP_SERVER_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.mcp.mcp_server")
        registry = mod.MCP_SERVER_REGISTRY
    except Exception:
        yield
        return

    snap = _snapshot(registry)
    yield
    _restore(registry, snap)


# ---------------------------------------------------------------------------
# MCP_TOOL_SCHEMA_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_mcp_tool_schema_registry():
    """Snapshot/restore MCP_TOOL_SCHEMA_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.mcp.tool_schema_registry")
        registry = mod.MCP_TOOL_SCHEMA_REGISTRY
    except Exception:
        yield
        return

    snap = _snapshot(registry)
    yield
    _restore(registry, snap)


# ---------------------------------------------------------------------------
# SSE_SERVER_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_sse_server_registry():
    """Snapshot/restore SSE_SERVER_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.mcp.sse_mcp_server")
        registry = mod.SSE_SERVER_REGISTRY
    except Exception:
        yield
        return

    snap = _snapshot(registry)
    yield
    _restore(registry, snap)


# ---------------------------------------------------------------------------
# SAFE_EXTRACTOR_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_safe_extractor_registry():
    """Snapshot/restore SAFE_EXTRACTOR_REGISTRY around every test."""
    try:
        mod = importlib.import_module("src.security.safe_archive")
        registry = mod.SAFE_EXTRACTOR_REGISTRY
    except Exception:
        yield
        return

    snap = _snapshot(registry)
    yield
    _restore(registry, snap)
