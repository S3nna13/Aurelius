"""Tests for the function-calling API shape validator."""

from __future__ import annotations

import json

import pytest

from src.serving.function_calling_api import (
    ALLOWED_TYPES,
    DEFAULT_TOOL_CHOICE,
    FunctionCallError,
    FunctionCallValidator,
    FunctionSchema,
    ToolChoice,
    ToolDefinition,
)


def _simple_schema() -> FunctionSchema:
    return FunctionSchema(
        name="get_weather",
        description="Fetch current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["city"],
        },
    )


def test_valid_tool_definition_passes():
    v = FunctionCallValidator()
    td = ToolDefinition(function=_simple_schema())
    v.validate_tool_definition(td)  # must not raise


def test_bad_parameters_not_object_raises():
    v = FunctionCallValidator()
    bad = FunctionSchema(
        name="f",
        description="d",
        parameters={"type": "array", "properties": {}, "required": []},
    )
    with pytest.raises(FunctionCallError):
        v.validate_tool_definition(ToolDefinition(function=bad))


def test_required_field_missing_from_properties_raises():
    v = FunctionCallValidator()
    bad = FunctionSchema(
        name="f",
        description="d",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["b"],
        },
    )
    with pytest.raises(FunctionCallError):
        v.validate_tool_definition(ToolDefinition(function=bad))


def test_tool_call_malformed_arguments_raises():
    v = FunctionCallValidator()
    raw = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": "{not json"},
        }
    ]
    with pytest.raises(FunctionCallError):
        v.parse_tool_calls(raw)


def test_tool_choice_named_requires_name():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.validate_tool_choice(ToolChoice(mode="named"), ["a"])


def test_tool_choice_auto_needs_no_name():
    v = FunctionCallValidator()
    v.validate_tool_choice(ToolChoice(mode="auto"), ["a"])


def test_validate_tool_choice_rejects_unknown_named_tool():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.validate_tool_choice(ToolChoice(mode="named", name="missing"), ["a", "b"])


def test_parse_tool_calls_length_and_round_trip():
    v = FunctionCallValidator()
    args_obj = {"city": "Austin", "unit": "F"}
    raw = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps(args_obj),
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        },
    ]
    parsed = v.parse_tool_calls(raw)
    assert len(parsed) == 2
    assert parsed[0].id == "call_1"
    assert parsed[0].function_name == "get_weather"
    assert json.loads(parsed[0].arguments_json) == args_obj
    assert parsed[1].function_name == "ping"


def test_format_tool_message_shape():
    v = FunctionCallValidator()
    msg = v.format_tool_message("call_1", "get_weather", "72F sunny")
    assert msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_weather",
        "content": "72F sunny",
    }


def test_validate_function_schema_accepts_simple():
    v = FunctionCallValidator()
    v.validate_function_schema_json_compatible(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": [],
        }
    )


def test_validate_function_schema_rejects_non_dict():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.validate_function_schema_json_compatible("not a dict")  # type: ignore[arg-type]


def test_tool_definition_type_must_be_function():
    v = FunctionCallValidator()
    td = ToolDefinition(function=_simple_schema(), type="plugin")
    with pytest.raises(FunctionCallError):
        v.validate_tool_definition(td)


def test_default_tool_choice_mode_is_auto():
    assert DEFAULT_TOOL_CHOICE.mode == "auto"
    assert DEFAULT_TOOL_CHOICE.name is None
    assert ALLOWED_TYPES == ("function",)


def test_unicode_function_name_allowed():
    v = FunctionCallValidator()
    schema = FunctionSchema(
        name="搜索_weather_ñ",
        description="unicode ok",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    v.validate_tool_definition(ToolDefinition(function=schema))


def test_schema_property_without_type_is_accepted():
    """Permissive policy: property spec with no 'type' key is allowed
    (means any-JSON-value, matching OpenAI/Claude leniency)."""
    v = FunctionCallValidator()
    v.validate_function_schema_json_compatible(
        {
            "type": "object",
            "properties": {"freeform": {"description": "any value"}},
            "required": [],
        }
    )


def test_determinism_same_input_same_output():
    v = FunctionCallValidator()
    raw = [
        {
            "id": "c",
            "type": "function",
            "function": {"name": "f", "arguments": '{"k": 1}'},
        }
    ]
    a = v.parse_tool_calls(raw)
    b = v.parse_tool_calls(raw)
    assert [(x.id, x.function_name, x.arguments_json) for x in a] == [
        (x.id, x.function_name, x.arguments_json) for x in b
    ]
    # format_tool_message is deterministic too
    assert v.format_tool_message("i", "n", "c") == v.format_tool_message("i", "n", "c")


def test_tool_choice_non_named_forbids_name():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.validate_tool_choice(ToolChoice(mode="auto", name="x"), ["x"])


def test_tool_choice_unknown_mode_raises():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.validate_tool_choice(ToolChoice(mode="bogus"), [])


def test_parse_tool_calls_rejects_non_list():
    v = FunctionCallValidator()
    with pytest.raises(FunctionCallError):
        v.parse_tool_calls({"not": "a list"})  # type: ignore[arg-type]


def test_parse_tool_calls_rejects_non_string_arguments():
    v = FunctionCallValidator()
    raw = [
        {
            "id": "c",
            "type": "function",
            "function": {"name": "f", "arguments": {"already": "dict"}},
        }
    ]
    with pytest.raises(FunctionCallError):
        v.parse_tool_calls(raw)
