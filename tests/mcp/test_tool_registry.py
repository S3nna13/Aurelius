"""Tests for src.mcp.tool_registry.

Coverage (≥ 28 tests):
- Register a tool successfully
- get() returns existing tool
- get() returns None for missing tool
- Duplicate register raises ValueError
- validate_call with all required params present returns []
- validate_call with missing required params returns missing names
- validate_call with unknown tool returns ["tool_not_found"]
- validate_call optional params not flagged as missing
- list_tools returns sorted names
- unregister returns True when tool existed
- unregister returns False when tool absent
- unregister removes tool from registry
- to_schema returns list of dicts
- to_schema structure matches OpenAI format
- to_schema required params list
- to_schema optional params excluded from required list
- to_schema properties contain type and description
- to_schema is sorted by tool name
- TOOL_REGISTRY_REGISTRY contains "default" key
- TOOL_REGISTRY_REGISTRY["default"] is ToolRegistry
- Registry instances are independent
- Re-registering after unregister succeeds
"""

from __future__ import annotations

import pytest

from src.mcp.tool_registry import (
    TOOL_REGISTRY_REGISTRY,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search",
        description="Web search",
        parameters=[
            ToolParameter("query", "string", required=True, description="Search query"),
            ToolParameter("limit", "integer", required=False, description="Max results"),
        ],
    )


def _noop_tool() -> ToolDefinition:
    return ToolDefinition(
        name="noop",
        description="Does nothing",
        parameters=[],
    )


def _echo_tool() -> ToolDefinition:
    return ToolDefinition(
        name="echo",
        description="Echoes input",
        parameters=[
            ToolParameter("text", "string", required=True, description="Text to echo"),
        ],
    )


# ---------------------------------------------------------------------------
# Register / get
# ---------------------------------------------------------------------------


def test_register_and_get_returns_tool():
    reg = _make_registry()
    tool = _search_tool()
    reg.register(tool)
    assert reg.get("search") is tool


def test_get_missing_returns_none():
    reg = _make_registry()
    assert reg.get("nonexistent") is None


def test_duplicate_register_raises_value_error():
    reg = _make_registry()
    reg.register(_search_tool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_search_tool())


def test_register_multiple_tools():
    reg = _make_registry()
    reg.register(_search_tool())
    reg.register(_noop_tool())
    assert reg.get("search") is not None
    assert reg.get("noop") is not None


def test_tool_definition_version_default():
    tool = _search_tool()
    assert tool.version == "1.0.0"


def test_tool_parameter_frozen():
    param = ToolParameter("p", "string")
    with pytest.raises((AttributeError, TypeError)):
        param.name = "q"  # type: ignore[misc]


def test_tool_definition_frozen():
    tool = _noop_tool()
    with pytest.raises((AttributeError, TypeError)):
        tool.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_call
# ---------------------------------------------------------------------------


def test_validate_call_all_required_present_returns_empty():
    reg = _make_registry()
    reg.register(_search_tool())
    missing = reg.validate_call("search", {"query": "hello", "limit": 5})
    assert missing == []


def test_validate_call_missing_required_param():
    reg = _make_registry()
    reg.register(_search_tool())
    missing = reg.validate_call("search", {})
    assert "query" in missing


def test_validate_call_optional_not_in_missing():
    reg = _make_registry()
    reg.register(_search_tool())
    # "limit" is optional; only "query" is required
    missing = reg.validate_call("search", {"query": "hi"})
    assert missing == []
    assert "limit" not in missing


def test_validate_call_unknown_tool_returns_tool_not_found():
    reg = _make_registry()
    result = reg.validate_call("ghost", {})
    assert result == ["tool_not_found"]


def test_validate_call_no_params_tool_always_valid():
    reg = _make_registry()
    reg.register(_noop_tool())
    assert reg.validate_call("noop", {}) == []


def test_validate_call_reports_all_missing_required():
    reg = _make_registry()
    tool = ToolDefinition(
        name="multi",
        description="Multi param",
        parameters=[
            ToolParameter("a", "string", required=True),
            ToolParameter("b", "integer", required=True),
        ],
    )
    reg.register(tool)
    missing = reg.validate_call("multi", {})
    assert set(missing) == {"a", "b"}


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


def test_list_tools_empty_registry():
    reg = _make_registry()
    assert reg.list_tools() == []


def test_list_tools_returns_sorted_names():
    reg = _make_registry()
    reg.register(_search_tool())
    reg.register(_noop_tool())
    reg.register(_echo_tool())
    assert reg.list_tools() == ["echo", "noop", "search"]


def test_list_tools_is_list():
    reg = _make_registry()
    reg.register(_noop_tool())
    assert isinstance(reg.list_tools(), list)


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


def test_unregister_existing_returns_true():
    reg = _make_registry()
    reg.register(_noop_tool())
    assert reg.unregister("noop") is True


def test_unregister_removes_tool():
    reg = _make_registry()
    reg.register(_noop_tool())
    reg.unregister("noop")
    assert reg.get("noop") is None


def test_unregister_absent_returns_false():
    reg = _make_registry()
    assert reg.unregister("ghost") is False


def test_reregister_after_unregister_succeeds():
    reg = _make_registry()
    reg.register(_noop_tool())
    reg.unregister("noop")
    reg.register(_noop_tool())  # should not raise
    assert reg.get("noop") is not None


# ---------------------------------------------------------------------------
# to_schema
# ---------------------------------------------------------------------------


def test_to_schema_returns_list():
    reg = _make_registry()
    reg.register(_search_tool())
    schema = reg.to_schema()
    assert isinstance(schema, list)


def test_to_schema_length_matches_registered_tools():
    reg = _make_registry()
    reg.register(_search_tool())
    reg.register(_noop_tool())
    assert len(reg.to_schema()) == 2


def test_to_schema_structure():
    reg = _make_registry()
    reg.register(_noop_tool())
    schema = reg.to_schema()
    entry = schema[0]
    assert "name" in entry
    assert "description" in entry
    assert "parameters" in entry
    params = entry["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params


def test_to_schema_required_params():
    reg = _make_registry()
    reg.register(_search_tool())
    schema = reg.to_schema()
    entry = next(e for e in schema if e["name"] == "search")
    assert "query" in entry["parameters"]["required"]
    assert "limit" not in entry["parameters"]["required"]


def test_to_schema_properties_contain_type_and_description():
    reg = _make_registry()
    reg.register(_search_tool())
    schema = reg.to_schema()
    entry = next(e for e in schema if e["name"] == "search")
    props = entry["parameters"]["properties"]
    assert props["query"]["type"] == "string"
    assert "description" in props["query"]


def test_to_schema_sorted_by_tool_name():
    reg = _make_registry()
    reg.register(_search_tool())
    reg.register(_noop_tool())
    reg.register(_echo_tool())
    names = [e["name"] for e in reg.to_schema()]
    assert names == sorted(names)


def test_to_schema_empty_registry_returns_empty_list():
    reg = _make_registry()
    assert reg.to_schema() == []


# ---------------------------------------------------------------------------
# Registry constant
# ---------------------------------------------------------------------------


def test_tool_registry_registry_contains_default():
    assert "default" in TOOL_REGISTRY_REGISTRY


def test_tool_registry_registry_default_is_tool_registry_class():
    assert TOOL_REGISTRY_REGISTRY["default"] is ToolRegistry


def test_registry_instances_are_independent():
    r1 = ToolRegistry()
    r2 = ToolRegistry()
    r1.register(_noop_tool())
    assert r2.get("noop") is None
