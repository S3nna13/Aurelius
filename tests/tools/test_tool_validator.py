from __future__ import annotations

import pytest

from src.tools.tool_schema_registry import ParameterSchema, ToolSchema, ToolSchemaRegistry
from src.tools.tool_validator import ToolCallValidator, ValidationError, ValidationResult


def _registry_with_search() -> ToolSchemaRegistry:
    reg = ToolSchemaRegistry()
    reg.register(
        ToolSchema(
            name="search",
            description="Web search",
            parameters=[
                ParameterSchema(name="query", type="string"),
                ParameterSchema(name="limit", type="integer", required=False, default=10),
                ParameterSchema(name="verbose", type="boolean", required=False, default=False),
            ],
        )
    )
    return reg


def _registry_with_enum() -> ToolSchemaRegistry:
    reg = ToolSchemaRegistry()
    reg.register(
        ToolSchema(
            name="mode_tool",
            description="Set mode",
            parameters=[
                ParameterSchema(name="mode", type="string", enum=["fast", "slow", "auto"]),
            ],
        )
    )
    return reg


class TestValidationResult:
    def test_dataclass_fields(self):
        r = ValidationResult(valid=True, errors=[], warnings=[], tool_name="t", call_args={})
        assert r.valid is True
        assert r.tool_name == "t"


class TestValidate:
    def test_valid_call(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {"query": "hello"})
        assert result.valid is True
        assert result.errors == []

    def test_missing_required_param(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {})
        assert result.valid is False
        assert any("query" in e for e in result.errors)

    def test_unknown_tool(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("ghost", {})
        assert result.valid is False
        assert any("unknown tool" in e for e in result.errors)

    def test_wrong_type_integer(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {"query": "hi", "limit": "ten"})
        assert result.valid is False
        assert any("limit" in e for e in result.errors)

    def test_wrong_type_string_given_int(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {"query": 123})
        assert result.valid is False

    def test_missing_optional_produces_warning(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {"query": "hi"})
        assert result.valid is True
        assert any("limit" in w for w in result.warnings)

    def test_enum_valid_value(self):
        v = ToolCallValidator(_registry_with_enum())
        result = v.validate("mode_tool", {"mode": "fast"})
        assert result.valid is True

    def test_enum_invalid_value(self):
        v = ToolCallValidator(_registry_with_enum())
        result = v.validate("mode_tool", {"mode": "turbo"})
        assert result.valid is False
        assert any("mode" in e for e in result.errors)

    def test_float_accepts_int(self):
        reg = ToolSchemaRegistry()
        reg.register(
            ToolSchema(
                name="calc",
                description="calc",
                parameters=[ParameterSchema(name="val", type="float")],
            )
        )
        v = ToolCallValidator(reg)
        result = v.validate("calc", {"val": 5})
        assert result.valid is True

    def test_tool_name_in_result(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.validate("search", {"query": "x"})
        assert result.tool_name == "search"

    def test_call_args_preserved(self):
        v = ToolCallValidator(_registry_with_search())
        args = {"query": "x"}
        result = v.validate("search", args)
        assert result.call_args == args


class TestValidateStrict:
    def test_returns_args_on_success(self):
        v = ToolCallValidator(_registry_with_search())
        args = {"query": "hello"}
        returned = v.validate_strict("search", args)
        assert returned == args

    def test_raises_on_failure(self):
        v = ToolCallValidator(_registry_with_search())
        with pytest.raises(ValidationError):
            v.validate_strict("search", {})

    def test_error_message_contains_detail(self):
        v = ToolCallValidator(_registry_with_search())
        with pytest.raises(ValidationError, match="query"):
            v.validate_strict("search", {})


class TestCoerce:
    def test_string_to_int(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.coerce("search", {"query": "hi", "limit": "42"})
        assert result["limit"] == 42
        assert isinstance(result["limit"], int)

    def test_string_true_to_bool(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.coerce("search", {"query": "hi", "verbose": "true"})
        assert result["verbose"] is True

    def test_string_false_to_bool(self):
        v = ToolCallValidator(_registry_with_search())
        result = v.coerce("search", {"query": "hi", "verbose": "false"})
        assert result["verbose"] is False

    def test_unknown_tool_returns_copy(self):
        v = ToolCallValidator(_registry_with_search())
        args = {"x": 1}
        result = v.coerce("nope", args)
        assert result == args

    def test_original_not_mutated(self):
        v = ToolCallValidator(_registry_with_search())
        args = {"query": "hi", "limit": "5"}
        v.coerce("search", args)
        assert args["limit"] == "5"

    def test_int_to_float(self):
        reg = ToolSchemaRegistry()
        reg.register(
            ToolSchema(
                name="calc",
                description="",
                parameters=[ParameterSchema(name="val", type="float")],
            )
        )
        v = ToolCallValidator(reg)
        result = v.coerce("calc", {"val": "3.14"})
        assert abs(result["val"] - 3.14) < 1e-9

    def test_non_string_to_string(self):
        reg = ToolSchemaRegistry()
        reg.register(
            ToolSchema(
                name="fmt",
                description="",
                parameters=[ParameterSchema(name="label", type="string")],
            )
        )
        v = ToolCallValidator(reg)
        result = v.coerce("fmt", {"label": 99})
        assert result["label"] == "99"


class TestBatchValidate:
    def test_returns_list(self):
        v = ToolCallValidator(_registry_with_search())
        results = v.batch_validate([("search", {"query": "hi"}), ("search", {})])
        assert len(results) == 2

    def test_individual_results_correct(self):
        v = ToolCallValidator(_registry_with_search())
        results = v.batch_validate([("search", {"query": "hi"}), ("search", {})])
        assert results[0].valid is True
        assert results[1].valid is False

    def test_empty_batch(self):
        v = ToolCallValidator(_registry_with_search())
        assert v.batch_validate([]) == []

    def test_uses_global_registry_by_default(self):
        from src.tools.tool_schema_registry import TOOL_SCHEMA_REGISTRY
        from src.tools.tool_validator import ToolCallValidator as TCV

        v = TCV()
        assert v._registry is TOOL_SCHEMA_REGISTRY
