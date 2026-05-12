"""Simple JSON validator with schema enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JSONSchema:
    required_fields: list[str] | None = None
    field_types: dict[str, type] | None = None
    allow_extra: bool = True


@dataclass
class JSONValidator:
    """Validate JSON data against a schema."""

    def validate(self, data: dict[str, Any], schema: JSONSchema) -> tuple[bool, list[str]]:
        errors = []
        if schema.required_fields:
            for f in schema.required_fields:
                if f not in data:
                    errors.append(f"missing required field: {f}")
        if schema.field_types:
            for f, t in schema.field_types.items():
                if f in data and not isinstance(data[f], t):
                    errors.append(
                        f"field '{f}' expected {t.__name__}, got {type(data[f]).__name__}"
                    )
        return len(errors) == 0, errors


JSON_VALIDATOR = JSONValidator()
