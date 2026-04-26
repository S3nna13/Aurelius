"""Tests for src/inference/tool_use.py"""

from __future__ import annotations

import json

import pytest
import torch

from src.inference.tool_use import (
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    ToolUseSession,
    format_tool_result_for_context,
    format_tools_for_prompt,
    parse_tool_call,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def model(small_config):
    torch.manual_seed(0)
    m = AureliusTransformer(small_config)
    m.eval()
    return m


def tokenize(text: str) -> list[int]:
    """Byte-level tokenizer capped to avoid seq-len overflow."""
    return list(text.encode("utf-8", errors="replace"))[:200]


@pytest.fixture
def weather_schema():
    return ToolSchema(
        name="get_weather",
        description="Fetch current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
        },
        required=["city"],
    )


@pytest.fixture
def add_schema():
    return ToolSchema(
        name="add",
        description="Add two numbers together.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        required=["a", "b"],
    )


@pytest.fixture
def registry(weather_schema, add_schema):
    reg = ToolRegistry()
    reg.register(weather_schema, lambda city: f"Sunny, 22 degrees in {city}")
    reg.register(add_schema, lambda a, b: str(a + b))
    return reg


@pytest.fixture
def session(model, registry):
    return ToolUseSession(
        model=model,
        tokenize_fn=tokenize,
        registry=registry,
        max_new_tokens=16,
        max_tool_rounds=3,
    )


# ---------------------------------------------------------------------------
# 1. test_tool_schema_to_prompt_str
# ---------------------------------------------------------------------------


def test_tool_schema_to_prompt_str(weather_schema):
    prompt_str = weather_schema.to_prompt_str()
    assert "get_weather" in prompt_str
    assert "Fetch current weather" in prompt_str


# ---------------------------------------------------------------------------
# 2. test_tool_call_to_json
# ---------------------------------------------------------------------------


def test_tool_call_to_json():
    call = ToolCall(tool_name="add", arguments={"a": 1, "b": 2}, call_id="abc123")
    result = call.to_json()
    parsed = json.loads(result)  # must be valid JSON
    assert parsed["name"] == "add"
    assert parsed["arguments"] == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# 3. test_tool_result_to_prompt_str
# ---------------------------------------------------------------------------


def test_tool_result_to_prompt_str():
    tr = ToolResult(call_id="xyz", tool_name="get_weather", result="Sunny in Paris")
    prompt_str = tr.to_prompt_str()
    assert "Sunny in Paris" in prompt_str


# ---------------------------------------------------------------------------
# 4. test_parse_tool_call_valid
# ---------------------------------------------------------------------------


def test_parse_tool_call_valid():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    call = parse_tool_call(text)
    assert call is not None
    assert call.tool_name == "get_weather"
    assert call.arguments == {"city": "Paris"}


# ---------------------------------------------------------------------------
# 5. test_parse_tool_call_invalid
# ---------------------------------------------------------------------------


def test_parse_tool_call_invalid():
    text = "This is just some random model output with no tool call at all."
    result = parse_tool_call(text)
    assert result is None


# ---------------------------------------------------------------------------
# 6. test_parse_tool_call_malformed_json
# ---------------------------------------------------------------------------


def test_parse_tool_call_malformed_json():
    text = "<tool_call>{not valid json!!!}</tool_call>"
    result = parse_tool_call(text)
    assert result is None


# ---------------------------------------------------------------------------
# 7. test_format_tools_for_prompt
# ---------------------------------------------------------------------------


def test_format_tools_for_prompt(weather_schema, add_schema):
    formatted = format_tools_for_prompt([weather_schema, add_schema])
    assert "get_weather" in formatted
    assert "add" in formatted


# ---------------------------------------------------------------------------
# 8. test_tool_registry_register_and_get
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_get(weather_schema):
    reg = ToolRegistry()
    reg.register(weather_schema, lambda city: f"Rainy in {city}")
    schema = reg.get_schema("get_weather")
    assert schema is not None
    assert schema.name == "get_weather"
    assert reg.get_schema("nonexistent") is None


# ---------------------------------------------------------------------------
# 9. test_tool_registry_execute_success
# ---------------------------------------------------------------------------


def test_tool_registry_execute_success(registry):
    call = ToolCall(tool_name="add", arguments={"a": 3, "b": 4}, call_id="test-001")
    result = registry.execute(call)
    assert isinstance(result, ToolResult)
    assert result.error is None
    assert result.result == "7"
    assert result.call_id == "test-001"
    assert result.tool_name == "add"


# ---------------------------------------------------------------------------
# 10. test_tool_registry_execute_error
# ---------------------------------------------------------------------------


def test_tool_registry_execute_error():
    """Exception in tool fn -> ToolResult with error set."""
    schema = ToolSchema(name="boom", description="Raises an error.", parameters={})
    reg = ToolRegistry()

    def boom_fn(**kwargs):
        raise ValueError("intentional failure")

    reg.register(schema, boom_fn)
    call = ToolCall(tool_name="boom", arguments={}, call_id="err-001")
    result = reg.execute(call)
    assert result.error is not None
    assert "intentional failure" in result.error
    assert result.result == ""


# ---------------------------------------------------------------------------
# 11. test_tool_use_session_run_no_tool_call
# ---------------------------------------------------------------------------


def test_tool_use_session_run_no_tool_call(session):
    """Session should return a string even when model never emits a tool call."""
    response = session.run("Hello, how are you?")
    assert isinstance(response, str)
    # The model output might be empty or garbled bytes but must not raise


# ---------------------------------------------------------------------------
# 12. test_format_tool_result_for_context
# ---------------------------------------------------------------------------


def test_format_tool_result_for_context():
    tr = ToolResult(call_id="call-42", tool_name="get_weather", result="Cloudy")
    formatted = format_tool_result_for_context(tr)
    assert 'call_id="call-42"' in formatted
    assert "Cloudy" in formatted
    assert formatted.startswith("<tool_result")
    assert formatted.endswith("</tool_result>")
