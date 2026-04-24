from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.tools.tool_schema_registry import (
    ParameterSchema,
    ToolSchemaRegistry,
    TOOL_SCHEMA_REGISTRY,
)

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    tool_name: str
    call_args: dict


class ToolCallValidator:
    """Validate tool call arguments against registered schema."""

    def __init__(self, registry: ToolSchemaRegistry | None = None) -> None:
        self._registry = registry if registry is not None else TOOL_SCHEMA_REGISTRY

    def validate(self, tool_name: str, args: dict) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        schema = self._registry.get(tool_name)
        if schema is None:
            errors.append(f"unknown tool: {tool_name!r}")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                tool_name=tool_name,
                call_args=args,
            )

        for param in schema.parameters:
            if param.name not in args:
                if param.required:
                    errors.append(f"missing required parameter: {param.name!r}")
                else:
                    warnings.append(f"missing optional parameter: {param.name!r}")
                continue

            value = args[param.name]
            type_error = self._check_type(param, value)
            if type_error:
                errors.append(type_error)
                continue

            if param.enum is not None and value not in param.enum:
                errors.append(
                    f"parameter {param.name!r} value {value!r} not in allowed values: {param.enum}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_name=tool_name,
            call_args=args,
        )

    def validate_strict(self, tool_name: str, args: dict) -> dict:
        result = self.validate(tool_name, args)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))
        return args

    def coerce(self, tool_name: str, args: dict) -> dict:
        schema = self._registry.get(tool_name)
        if schema is None:
            return dict(args)

        coerced = dict(args)
        for param in schema.parameters:
            if param.name not in coerced:
                continue
            value = coerced[param.name]
            coerced[param.name] = self._coerce_value(param.type, value)
        return coerced

    def batch_validate(self, calls: list[tuple[str, dict]]) -> list[ValidationResult]:
        return [self.validate(name, args) for name, args in calls]

    def _check_type(self, param: ParameterSchema, value: Any) -> str:
        expected = _TYPE_MAP.get(param.type)
        if expected is None:
            return ""
        if param.type == "float" and isinstance(value, int):
            return ""
        if not isinstance(value, expected):
            return (
                f"parameter {param.name!r} expected type {param.type!r}, "
                f"got {type(value).__name__!r}"
            )
        return ""

    def _coerce_value(self, type_name: str, value: Any) -> Any:
        if type_name == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if type_name == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if type_name == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True
                if value.lower() in ("false", "0", "no"):
                    return False
            try:
                return bool(int(value))
            except (ValueError, TypeError):
                return value
        if type_name == "string":
            if not isinstance(value, str):
                return str(value)
        return value
