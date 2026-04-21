"""Integration: function-calling validator wiring in src.serving + config."""

from __future__ import annotations


def test_function_calling_api_registered_in_api_shape_registry():
    from src.serving import API_SHAPE_REGISTRY, FunctionCallValidator

    assert "function_calling.validator" in API_SHAPE_REGISTRY
    assert API_SHAPE_REGISTRY["function_calling.validator"] is (
        FunctionCallValidator
    )


def test_function_calling_symbols_exported_from_serving():
    from src.serving import (
        DEFAULT_TOOL_CHOICE,
        FunctionCallError,
        FunctionCallValidator,
        FunctionSchema,
        ToolCall,
        ToolChoice,
        ToolDefinition,
    )

    assert DEFAULT_TOOL_CHOICE.mode == "auto"
    assert issubclass(FunctionCallError, Exception)
    # All classes are callable/constructable sentinels
    assert FunctionCallValidator is not None
    assert FunctionSchema is not None
    assert ToolCall is not None
    assert ToolChoice is not None
    assert ToolDefinition is not None


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert cfg.serving_function_calling_api_enabled is False


def test_end_to_end_happy_path_via_registry():
    from src.serving import (
        API_SHAPE_REGISTRY,
        FunctionSchema,
        ToolChoice,
        ToolDefinition,
    )

    cls = API_SHAPE_REGISTRY["function_calling.validator"]
    v = cls()
    schema = FunctionSchema(
        name="lookup",
        description="look up a thing",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
    v.validate_tool_definition(ToolDefinition(function=schema))
    v.validate_tool_choice(ToolChoice(mode="named", name="lookup"), ["lookup"])
    parsed = v.parse_tool_calls(
        [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"},
            }
        ]
    )
    assert parsed[0].function_name == "lookup"
