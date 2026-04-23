"""Tests for src.mcp.tool_schema_registry.

Coverage:
- register_tool_schema -> retrievable via get_tool_schema
- validate_tool_call passes for correct args, fails for missing required keys
- Schema version mismatch raises ToolSchemaError
- Malformed tool manifest (missing name) raises ToolSchemaError
"""

from __future__ import annotations

import pytest

from src.mcp.tool_schema_registry import (
    MCP_TOOL_SCHEMA_REGISTRY,
    ToolSchema,
    ToolSchemaError,
    get_tool_schema,
    register_tool_schema,
    validate_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name: str = "test_tool", version: str = "1.0") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="A test tool.",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x"],
        },
        output_schema={"type": "object"},
        version=version,
        tags=["test"],
    )


# ---------------------------------------------------------------------------
# ToolSchema dataclass validation
# ---------------------------------------------------------------------------


def test_missing_name_raises_tool_schema_error():
    with pytest.raises(ToolSchemaError, match="name must be a non-empty string"):
        ToolSchema(name="", description="oops")


def test_whitespace_only_name_raises_tool_schema_error():
    with pytest.raises(ToolSchemaError, match="name must be a non-empty string"):
        ToolSchema(name="   ", description="oops")


def test_invalid_version_raises_tool_schema_error():
    with pytest.raises(ToolSchemaError, match="version must match"):
        ToolSchema(name="x", description="d", version="v1")


def test_non_dict_input_schema_raises():
    with pytest.raises(ToolSchemaError, match="input_schema must be a dict"):
        ToolSchema(name="x", description="d", input_schema="not a dict")  # type: ignore


def test_non_list_tags_raises():
    with pytest.raises(ToolSchemaError, match="tags must be a list"):
        ToolSchema(name="x", description="d", tags="tag1")  # type: ignore


# ---------------------------------------------------------------------------
# Registration and retrieval
# ---------------------------------------------------------------------------


def test_register_and_retrieve(tmp_registry):
    schema = _make_schema("reg_test_tool")
    register_tool_schema(schema)
    retrieved = get_tool_schema("reg_test_tool")
    assert retrieved is schema


def test_get_tool_schema_unknown_raises():
    with pytest.raises(KeyError, match="No tool schema registered"):
        get_tool_schema("__absolutely_nonexistent__")


def test_register_non_schema_raises():
    with pytest.raises(ToolSchemaError):
        register_tool_schema({"name": "bad"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


def test_compatible_version_update_allowed(tmp_registry):
    schema_v1 = _make_schema("versioned_tool", version="1.0")
    schema_v1_1 = _make_schema("versioned_tool", version="1.1")
    register_tool_schema(schema_v1)
    # Should not raise — same major version
    register_tool_schema(schema_v1_1)
    assert get_tool_schema("versioned_tool").version == "1.1"


def test_incompatible_version_raises_tool_schema_error(tmp_registry):
    schema_v1 = _make_schema("conflict_tool", version="1.0")
    schema_v2 = _make_schema("conflict_tool", version="2.0")
    register_tool_schema(schema_v1)
    with pytest.raises(ToolSchemaError, match="Version mismatch"):
        register_tool_schema(schema_v2)


# ---------------------------------------------------------------------------
# validate_tool_call
# ---------------------------------------------------------------------------


def test_validate_tool_call_passes_for_correct_args(tmp_registry):
    schema = _make_schema("validate_test_tool")
    register_tool_schema(schema)
    result = validate_tool_call("validate_test_tool", {"x": 10})
    assert result is True


def test_validate_tool_call_passes_with_extra_args(tmp_registry):
    schema = _make_schema("validate_extra_tool")
    register_tool_schema(schema)
    result = validate_tool_call("validate_extra_tool", {"x": 1, "extra": "ignored"})
    assert result is True


def test_validate_tool_call_fails_for_missing_required(tmp_registry):
    schema = _make_schema("validate_fail_tool")
    register_tool_schema(schema)
    with pytest.raises(ToolSchemaError, match="missing required argument"):
        validate_tool_call("validate_fail_tool", {"y": 5})  # 'x' missing


def test_validate_tool_call_unknown_tool_raises():
    with pytest.raises(KeyError):
        validate_tool_call("__unknown_validate_tool__", {"x": 1})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_registry():
    """Snapshot and restore the registry around a test."""
    snapshot = dict(MCP_TOOL_SCHEMA_REGISTRY)
    yield
    MCP_TOOL_SCHEMA_REGISTRY.clear()
    MCP_TOOL_SCHEMA_REGISTRY.update(snapshot)
