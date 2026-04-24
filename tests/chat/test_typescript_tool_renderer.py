"""Tests for src/chat/typescript_tool_renderer.py."""

from __future__ import annotations

import pytest

from src.chat.typescript_tool_renderer import (
    ts_type,
    render_function_signature,
    render_namespace,
    render_tools_markdown,
    TypeScriptToolRenderer,
    TS_TOOL_RENDERER,
)


# ---------------------------------------------------------------------------
# ts_type
# ---------------------------------------------------------------------------


class TestTsType:
    def test_string_type(self):
        assert ts_type({"type": "string"}) == "string"

    def test_integer_type(self):
        assert ts_type({"type": "integer"}) == "number"

    def test_number_type(self):
        assert ts_type({"type": "number"}) == "number"

    def test_boolean_type(self):
        assert ts_type({"type": "boolean"}) == "boolean"

    def test_array_of_strings(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert ts_type(schema) == "string[]"

    def test_array_of_numbers(self):
        schema = {"type": "array", "items": {"type": "number"}}
        assert ts_type(schema) == "number[]"

    def test_enum_type(self):
        schema = {"type": "string", "enum": ["a", "b", "c"]}
        result = ts_type(schema)
        assert '"a"' in result
        assert '"b"' in result
        assert '"c"' in result
        assert "|" in result

    def test_object_type(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        result = ts_type(schema)
        assert "name" in result
        assert "age" in result
        assert "string" in result
        assert "number" in result

    def test_nullable_when_not_required(self):
        schema = {"type": "string"}
        result = ts_type(schema, required_fields=["other_field"], field_name="my_field")
        assert "| null" in result

    def test_not_nullable_when_required(self):
        schema = {"type": "string"}
        result = ts_type(schema, required_fields=["my_field"], field_name="my_field")
        assert "| null" not in result

    def test_unknown_type(self):
        assert ts_type({}) == "unknown"

    def test_empty_object_schema(self):
        schema = {"type": "object", "properties": {}}
        result = ts_type(schema)
        assert "Record<string, unknown>" in result


# ---------------------------------------------------------------------------
# render_function_signature
# ---------------------------------------------------------------------------


class TestRenderFunctionSignature:
    def test_basic_function(self):
        tool = {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        result = render_function_signature(tool)
        assert "type search = " in result
        assert "query" in result
        assert "=> any;" in result

    def test_no_parameters(self):
        tool = {"name": "ping", "description": "Ping", "parameters": {}}
        result = render_function_signature(tool)
        assert "type ping = " in result
        assert "{}" in result

    def test_optional_parameter(self):
        tool = {
            "name": "foo",
            "parameters": {
                "type": "object",
                "properties": {"opt": {"type": "string"}},
                "required": [],
            },
        }
        result = render_function_signature(tool)
        # Optional parameter should have "?" suffix
        assert "opt?" in result


# ---------------------------------------------------------------------------
# render_namespace
# ---------------------------------------------------------------------------


class TestRenderNamespace:
    def test_namespace_block_structure(self):
        tools = [
            {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = render_namespace("functions", tools)
        assert "## functions" in result
        assert "namespace functions {" in result
        assert "search" in result
        assert result.endswith("}")

    def test_multiple_tools(self):
        tools = [
            {"name": "search", "description": "S", "parameters": {}},
            {"name": "calc", "description": "C", "parameters": {}},
        ]
        result = render_namespace("functions", tools)
        assert "search" in result
        assert "calc" in result

    def test_description_as_comment(self):
        tools = [{"name": "foo", "description": "Does stuff", "parameters": {}}]
        result = render_namespace("functions", tools)
        assert "// Does stuff" in result


# ---------------------------------------------------------------------------
# render_tools_markdown
# ---------------------------------------------------------------------------


class TestRenderToolsMarkdown:
    def test_starts_with_functions_header(self):
        tools = [{"name": "foo", "description": "Bar", "parameters": {}}]
        result = render_tools_markdown(tools)
        assert result.startswith("## Functions\n\n")

    def test_contains_function_signatures(self):
        tools = [{"name": "bar", "parameters": {}}]
        result = render_tools_markdown(tools)
        assert "type bar = " in result


# ---------------------------------------------------------------------------
# TypeScriptToolRenderer class and singleton
# ---------------------------------------------------------------------------


class TestTypeScriptToolRendererClass:
    def test_singleton_instance(self):
        assert isinstance(TS_TOOL_RENDERER, TypeScriptToolRenderer)

    def test_ts_type_via_class(self):
        assert TS_TOOL_RENDERER.ts_type({"type": "string"}) == "string"

    def test_render_function_signature_via_class(self):
        tool = {"name": "x", "parameters": {}}
        result = TS_TOOL_RENDERER.render_function_signature(tool)
        assert "type x = " in result

    def test_render_namespace_via_class(self):
        tools = [{"name": "y", "parameters": {}}]
        result = TS_TOOL_RENDERER.render_namespace("ns", tools)
        assert "namespace ns {" in result

    def test_render_tools_markdown_via_class(self):
        tools = [{"name": "z", "parameters": {}}]
        result = TS_TOOL_RENDERER.render_tools_markdown(tools)
        assert "## Functions" in result
