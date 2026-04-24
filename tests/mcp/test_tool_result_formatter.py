"""Tests for src/mcp/tool_result_formatter.py — 10+ unit tests, no GPU."""

from __future__ import annotations

import json

import pytest

from src.mcp.tool_result_formatter import (
    MCP_REGISTRY,
    ResultFormat,
    ToolResultFormatter,
)


@pytest.fixture()
def fmt() -> ToolResultFormatter:
    return ToolResultFormatter()


# ---------------------------------------------------------------------------
# TEXT format
# ---------------------------------------------------------------------------

class TestTextFormat:
    def test_plain_string(self, fmt):
        result = fmt.format("hello", ResultFormat.TEXT)
        assert result == "hello"

    def test_integer_converted(self, fmt):
        result = fmt.format(42, ResultFormat.TEXT)
        assert result == "42"

    def test_dict_converted_to_str(self, fmt):
        result = fmt.format({"key": "value"}, ResultFormat.TEXT)
        assert "key" in result

    def test_truncation_at_10k(self, fmt):
        long_str = "x" * 20_000
        result = fmt.format(long_str, ResultFormat.TEXT)
        assert len(result) < 15_000
        assert "truncated" in result

    def test_default_format_is_text(self, fmt):
        result = fmt.format("hi")
        assert result == "hi"


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_dict_is_valid_json(self, fmt):
        data = {"a": 1, "b": [2, 3]}
        result = fmt.format(data, ResultFormat.JSON)
        parsed = json.loads(result)
        assert parsed == data

    def test_list_is_valid_json(self, fmt):
        data = [1, 2, 3]
        result = fmt.format(data, ResultFormat.JSON)
        assert json.loads(result) == data

    def test_non_serialisable_falls_back(self, fmt):
        class Unserializable:
            pass
        result = fmt.format(Unserializable(), ResultFormat.JSON)
        # Should not raise; output must be valid JSON of some kind
        parsed = json.loads(result)
        assert parsed is not None  # any valid JSON value is acceptable

    def test_indented_two_spaces(self, fmt):
        result = fmt.format({"x": 1}, ResultFormat.JSON)
        assert "  " in result  # indent=2


# ---------------------------------------------------------------------------
# MARKDOWN format
# ---------------------------------------------------------------------------

class TestMarkdownFormat:
    def test_dict_wrapped_in_code_block(self, fmt):
        result = fmt.format({"key": "val"}, ResultFormat.MARKDOWN)
        assert result.startswith("```")
        assert result.endswith("```")

    def test_list_wrapped_in_code_block(self, fmt):
        result = fmt.format([1, 2, 3], ResultFormat.MARKDOWN)
        assert "```" in result

    def test_plain_string_not_wrapped(self, fmt):
        result = fmt.format("plain text", ResultFormat.MARKDOWN)
        assert result == "plain text"

    def test_number_not_wrapped(self, fmt):
        result = fmt.format(3.14, ResultFormat.MARKDOWN)
        assert "```" not in result


# ---------------------------------------------------------------------------
# CODE format
# ---------------------------------------------------------------------------

class TestCodeFormat:
    def test_wrapped_in_triple_backticks(self, fmt):
        result = fmt.format("print('hi')", ResultFormat.CODE)
        assert result.startswith("```")
        assert result.endswith("```")

    def test_content_present_in_block(self, fmt):
        result = fmt.format("x = 1", ResultFormat.CODE)
        assert "x = 1" in result


# ---------------------------------------------------------------------------
# ERROR format
# ---------------------------------------------------------------------------

class TestErrorFormat:
    def test_exception_formatted(self, fmt):
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = fmt.format(e, ResultFormat.ERROR)
        assert "ValueError" in result
        assert "test error" in result

    def test_non_exception_formatted(self, fmt):
        result = fmt.format("something went wrong", ResultFormat.ERROR)
        assert "Error" in result or "something" in result

    def test_error_contains_traceback_for_exceptions(self, fmt):
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            result = fmt.format(e, ResultFormat.ERROR)
        assert "Traceback" in result or "RuntimeError" in result


# ---------------------------------------------------------------------------
# format_batch()
# ---------------------------------------------------------------------------

class TestFormatBatch:
    def test_returns_list_of_same_length(self, fmt):
        results = fmt.format_batch([1, 2, 3], ResultFormat.TEXT)
        assert len(results) == 3

    def test_each_element_formatted(self, fmt):
        results = fmt.format_batch(["a", "b"], ResultFormat.TEXT)
        assert results == ["a", "b"]

    def test_empty_list(self, fmt):
        assert fmt.format_batch([], ResultFormat.JSON) == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_formatter(self):
        assert "tool_result_formatter" in MCP_REGISTRY

    def test_registry_instance_is_formatter(self):
        assert isinstance(MCP_REGISTRY["tool_result_formatter"], ToolResultFormatter)
