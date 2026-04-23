"""Tests for tool_registry: ToolSpec, ToolResult, ToolRegistry, TOOL_REGISTRY."""
from __future__ import annotations

import pytest

from src.tools.tool_registry import (
    ToolSpec,
    ToolResult,
    ToolRegistry,
    TOOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# ToolSpec tests
# ---------------------------------------------------------------------------

class TestToolSpec:
    def test_toolspec_has_name_field(self):
        spec = ToolSpec(name="test", description="desc", parameters={})
        assert spec.name == "test"

    def test_toolspec_has_description_field(self):
        spec = ToolSpec(name="test", description="my desc", parameters={})
        assert spec.description == "my desc"

    def test_toolspec_has_parameters_field(self):
        params = {"type": "object", "properties": {}}
        spec = ToolSpec(name="test", description="desc", parameters=params)
        assert spec.parameters == params

    def test_toolspec_required_defaults_to_empty_list(self):
        spec = ToolSpec(name="test", description="desc", parameters={})
        assert spec.required == []

    def test_toolspec_required_can_be_set(self):
        spec = ToolSpec(name="test", description="desc", parameters={}, required=["a", "b"])
        assert spec.required == ["a", "b"]

    def test_toolspec_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(ToolSpec)

    def test_toolspec_required_lists_are_independent(self):
        spec1 = ToolSpec(name="a", description="a", parameters={})
        spec2 = ToolSpec(name="b", description="b", parameters={})
        spec1.required.append("x")
        assert spec2.required == []


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_toolresult_has_tool_name(self):
        r = ToolResult(tool_name="shell", success=True, output="ok")
        assert r.tool_name == "shell"

    def test_toolresult_has_success(self):
        r = ToolResult(tool_name="shell", success=True, output="ok")
        assert r.success is True

    def test_toolresult_has_output(self):
        r = ToolResult(tool_name="shell", success=True, output="hello")
        assert r.output == "hello"

    def test_toolresult_error_defaults_to_empty_string(self):
        r = ToolResult(tool_name="shell", success=True, output="ok")
        assert r.error == ""

    def test_toolresult_error_can_be_set(self):
        r = ToolResult(tool_name="shell", success=False, output="", error="bad")
        assert r.error == "bad"

    def test_toolresult_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(ToolResult)

    def test_toolresult_success_false(self):
        r = ToolResult(tool_name="x", success=False, output="")
        assert r.success is False


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def _make_registry(self):
        return ToolRegistry()

    def _make_spec(self, name="mytool"):
        return ToolSpec(name=name, description="A test tool", parameters={"type": "object", "properties": {}})

    def test_register_and_get_spec_round_trip(self):
        reg = self._make_registry()
        spec = self._make_spec()
        reg.register(spec, lambda: ToolResult(tool_name="mytool", success=True, output="ok"))
        assert reg.get_spec("mytool") is spec

    def test_get_spec_returns_none_for_unknown(self):
        reg = self._make_registry()
        assert reg.get_spec("nonexistent") is None

    def test_invoke_successful_handler_returns_toolresult_success(self):
        reg = self._make_registry()
        spec = self._make_spec("echo")
        reg.register(spec, lambda: ToolResult(tool_name="echo", success=True, output="pong"))
        result = reg.invoke("echo")
        assert result.success is True

    def test_invoke_successful_handler_output(self):
        reg = self._make_registry()
        spec = self._make_spec("echo")
        reg.register(spec, lambda: ToolResult(tool_name="echo", success=True, output="pong"))
        result = reg.invoke("echo")
        assert result.output == "pong"

    def test_invoke_handler_raises_returns_failure(self):
        reg = self._make_registry()
        spec = self._make_spec("bad")
        reg.register(spec, lambda: (_ for _ in ()).throw(ValueError("boom")))
        result = reg.invoke("bad")
        assert result.success is False

    def test_invoke_handler_raises_error_contains_message(self):
        reg = self._make_registry()
        spec = self._make_spec("bad")

        def bad_handler():
            raise RuntimeError("something failed")

        reg.register(spec, bad_handler)
        result = reg.invoke("bad")
        assert "something failed" in result.error

    def test_invoke_unknown_tool_returns_failure(self):
        reg = self._make_registry()
        result = reg.invoke("no_such_tool")
        assert result.success is False

    def test_invoke_unknown_tool_error_message(self):
        reg = self._make_registry()
        result = reg.invoke("no_such_tool")
        assert "no_such_tool" in result.error or "unknown" in result.error.lower()

    def test_list_tools_returns_registered_names(self):
        reg = self._make_registry()
        spec_a = self._make_spec("tool_a")
        spec_b = self._make_spec("tool_b")
        reg.register(spec_a, lambda: None)
        reg.register(spec_b, lambda: None)
        names = reg.list_tools()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_list_tools_empty_registry(self):
        reg = self._make_registry()
        assert reg.list_tools() == []

    def test_list_tools_returns_list(self):
        reg = self._make_registry()
        assert isinstance(reg.list_tools(), list)

    def test_to_openai_format_returns_list(self):
        reg = self._make_registry()
        spec = self._make_spec("fn1")
        reg.register(spec, lambda: None)
        result = reg.to_openai_format()
        assert isinstance(result, list)

    def test_to_openai_format_type_is_function(self):
        reg = self._make_registry()
        spec = self._make_spec("fn1")
        reg.register(spec, lambda: None)
        result = reg.to_openai_format()
        assert result[0]["type"] == "function"

    def test_to_openai_format_function_name(self):
        reg = self._make_registry()
        spec = self._make_spec("fn1")
        reg.register(spec, lambda: None)
        result = reg.to_openai_format()
        assert result[0]["function"]["name"] == "fn1"

    def test_to_openai_format_function_description(self):
        reg = self._make_registry()
        spec = self._make_spec("fn1")
        reg.register(spec, lambda: None)
        result = reg.to_openai_format()
        assert result[0]["function"]["description"] == "A test tool"

    def test_to_openai_format_contains_parameters(self):
        reg = self._make_registry()
        spec = self._make_spec("fn1")
        reg.register(spec, lambda: None)
        result = reg.to_openai_format()
        assert "parameters" in result[0]["function"]

    def test_to_openai_format_empty_registry(self):
        reg = self._make_registry()
        assert reg.to_openai_format() == []

    def test_to_openai_format_multiple_tools(self):
        reg = self._make_registry()
        reg.register(self._make_spec("a"), lambda: None)
        reg.register(self._make_spec("b"), lambda: None)
        result = reg.to_openai_format()
        assert len(result) == 2

    def test_invoke_passes_kwargs_to_handler(self):
        reg = self._make_registry()
        spec = ToolSpec(name="adder", description="add", parameters={}, required=["x", "y"])
        reg.register(spec, lambda x, y: ToolResult(tool_name="adder", success=True, output=str(x + y)))
        result = reg.invoke("adder", x=3, y=4)
        assert result.output == "7"

    def test_invoke_wraps_non_toolresult_output(self):
        reg = self._make_registry()
        spec = self._make_spec("str_tool")
        reg.register(spec, lambda: "plain string")
        result = reg.invoke("str_tool")
        assert result.success is True
        assert result.output == "plain string"

    def test_register_overwrites_existing_name(self):
        reg = self._make_registry()
        spec1 = ToolSpec(name="dup", description="first", parameters={})
        spec2 = ToolSpec(name="dup", description="second", parameters={})
        reg.register(spec1, lambda: None)
        reg.register(spec2, lambda: None)
        assert reg.get_spec("dup").description == "second"


# ---------------------------------------------------------------------------
# TOOL_REGISTRY module-level instance
# ---------------------------------------------------------------------------

class TestToolRegistryInstance:
    def test_tool_registry_exists(self):
        assert TOOL_REGISTRY is not None

    def test_tool_registry_is_instance(self):
        assert isinstance(TOOL_REGISTRY, ToolRegistry)

    def test_tool_registry_has_shell_registered(self):
        # shell_tool registers into TOOL_REGISTRY on import
        from src.tools import SHELL_TOOL  # noqa: F401
        assert "shell" in TOOL_REGISTRY.list_tools()

    def test_tool_registry_has_file_registered(self):
        from src.tools import FILE_TOOL  # noqa: F401
        assert "file" in TOOL_REGISTRY.list_tools()
