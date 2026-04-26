"""Tests for src.mcp.tool_schema_validator.

Coverage (≥ 12 tests):
- Valid schema returns empty errors
- is_valid returns True for valid schema
- Non-dict schema is rejected
- Missing name returns error
- Empty name returns error
- Invalid name format returns error
- Missing description returns error
- Empty description returns error
- Missing parameters returns error
- Parameters missing properties returns error
- Parameters missing required returns error
- Bad property type returns error
- Circular $ref returns error
- Oversized schema returns error
- Registry contains default
"""

from __future__ import annotations

from src.mcp.tool_schema_validator import (
    DEFAULT_TOOL_SCHEMA_VALIDATOR,
    TOOL_SCHEMA_VALIDATOR_REGISTRY,
    ToolSchemaValidator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_schema() -> dict:
    return {
        "name": "search",
        "description": "Web search tool",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        },
    }


# ---------------------------------------------------------------------------
# Valid schema
# ---------------------------------------------------------------------------


def test_valid_schema_returns_empty_errors():
    validator = ToolSchemaValidator()
    errors = validator.validate(_valid_schema())
    assert errors == []


def test_is_valid_returns_true_for_valid_schema():
    validator = ToolSchemaValidator()
    assert validator.is_valid(_valid_schema()) is True


# ---------------------------------------------------------------------------
# Bad input types
# ---------------------------------------------------------------------------


def test_non_dict_schema_is_rejected():
    validator = ToolSchemaValidator()
    errors = validator.validate("not a dict")
    assert any("must be a dict" in e for e in errors)


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


def test_missing_name_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["name"]
    errors = validator.validate(schema)
    assert any("name" in e and "Missing" in e for e in errors)


def test_empty_name_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["name"] = ""
    errors = validator.validate(schema)
    assert any("name" in e and "non-empty" in e for e in errors)


def test_invalid_name_format_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["name"] = "123-invalid"
    errors = validator.validate(schema)
    assert any("name" in e and "match" in e for e in errors)


def test_non_string_name_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["name"] = 42
    errors = validator.validate(schema)
    assert any("name" in e and "string" in e for e in errors)


# ---------------------------------------------------------------------------
# Description validation
# ---------------------------------------------------------------------------


def test_missing_description_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["description"]
    errors = validator.validate(schema)
    assert any("description" in e and "Missing" in e for e in errors)


def test_empty_description_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["description"] = "   "
    errors = validator.validate(schema)
    assert any("description" in e and "non-empty" in e for e in errors)


def test_non_string_description_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["description"] = []
    errors = validator.validate(schema)
    assert any("description" in e and "string" in e for e in errors)


# ---------------------------------------------------------------------------
# Parameters validation
# ---------------------------------------------------------------------------


def test_missing_parameters_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["parameters"]
    errors = validator.validate(schema)
    assert any("parameters" in e and "Missing" in e for e in errors)


def test_parameters_missing_properties_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["parameters"]["properties"]
    errors = validator.validate(schema)
    assert any("properties" in e for e in errors)


def test_parameters_missing_required_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["parameters"]["required"]
    errors = validator.validate(schema)
    assert any("required" in e for e in errors)


def test_parameters_properties_not_dict_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["parameters"]["properties"] = "bad"
    errors = validator.validate(schema)
    assert any("properties" in e and "dict" in e for e in errors)


def test_parameters_required_not_list_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["parameters"]["required"] = "bad"
    errors = validator.validate(schema)
    assert any("required" in e and "list" in e for e in errors)


# ---------------------------------------------------------------------------
# Property type validation
# ---------------------------------------------------------------------------


def test_bad_property_type_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["parameters"]["properties"]["query"]["type"] = "float64"
    errors = validator.validate(schema)
    assert any("query" in e and "float64" in e for e in errors)


def test_property_missing_type_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    del schema["parameters"]["properties"]["query"]["type"]
    errors = validator.validate(schema)
    assert any("query" in e and "type" in e for e in errors)


def test_property_not_dict_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    schema["parameters"]["properties"]["query"] = "bad"
    errors = validator.validate(schema)
    assert any("query" in e and "dict" in e for e in errors)


# ---------------------------------------------------------------------------
# Circular $ref
# ---------------------------------------------------------------------------


def test_circular_ref_returns_error():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    inner: dict = {"$ref": "#", "nested": {}}
    inner["nested"] = inner  # type: ignore[assignment]
    schema["parameters"]["properties"]["self_ref"] = inner
    errors = validator.validate(schema)
    assert any("Circular $ref" in e for e in errors)


def test_circular_dict_without_ref_does_not_flag():
    validator = ToolSchemaValidator()
    schema = _valid_schema()
    inner: dict = {"nested": {}}
    inner["nested"] = inner  # type: ignore[assignment]
    schema["parameters"]["properties"]["self_ref"] = inner
    errors = validator.validate(schema)
    assert all("Circular $ref" not in e for e in errors)


# ---------------------------------------------------------------------------
# Schema size
# ---------------------------------------------------------------------------


def test_oversized_schema_returns_error():
    validator = ToolSchemaValidator()
    # Build a schema with > 10_000 keys by nesting many properties
    properties = {}
    for i in range(10_010):
        properties[f"prop_{i}"] = {"type": "string"}
    schema = {
        "name": "big_tool",
        "description": "A tool with too many keys",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": [],
        },
    }
    errors = validator.validate(schema)
    assert any("exceeds maximum" in e for e in errors)


def test_schema_at_limit_is_valid():
    validator = ToolSchemaValidator()
    # The base valid schema has a small number of keys.
    # Build one that stays under 10_000.
    properties = {}
    for i in range(100):
        properties[f"prop_{i}"] = {"type": "string"}
    schema = {
        "name": "medium_tool",
        "description": "A tool with many keys but under limit",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": [],
        },
    }
    errors = validator.validate(schema)
    assert errors == []


# ---------------------------------------------------------------------------
# Registry constant
# ---------------------------------------------------------------------------


def test_tool_schema_validator_registry_contains_default():
    assert "default" in TOOL_SCHEMA_VALIDATOR_REGISTRY


def test_tool_schema_validator_registry_default_is_instance():
    assert isinstance(TOOL_SCHEMA_VALIDATOR_REGISTRY["default"], ToolSchemaValidator)


def test_default_tool_schema_validator_is_instance():
    assert isinstance(DEFAULT_TOOL_SCHEMA_VALIDATOR, ToolSchemaValidator)
