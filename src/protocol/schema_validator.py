"""Schema validator for the Aurelius protocol surface.

Provides lightweight, stdlib-only message schema definition and validation
without importing Pydantic, Marshmallow, or any other external library.

Supported type names: "str", "int", "float", "bool", "list", "dict".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SchemaType(StrEnum):
    JSON_SCHEMA = "json_schema"
    PYDANTIC_LIKE = "pydantic_like"
    PROTOBUF_LITE = "protobuf_lite"


@dataclass
class ValidationError:
    field: str
    message: str
    value: Any


@dataclass
class SchemaField:
    name: str
    type_name: str
    required: bool = True
    default: Any = None


@dataclass
class MessageSchema:
    name: str
    fields: list[SchemaField]
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Type-check mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def _check_type(value: Any, type_name: str) -> bool:
    """Return ``True`` if *value* matches *type_name*."""
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    expected = _TYPE_MAP.get(type_name)
    if expected is None:
        # Unknown type — accept anything
        return True
    return isinstance(value, expected)


class SchemaValidator:
    """Register schemas and validate plain dicts against them."""

    def __init__(self) -> None:
        self._schemas: dict[str, MessageSchema] = {}

    def define_schema(self, schema: MessageSchema) -> None:
        """Register *schema* by its ``name`` attribute."""
        self._schemas[schema.name] = schema

    def validate(
        self,
        data: dict,
        schema_name: str,
    ) -> list[ValidationError]:
        """Validate *data* against the named schema.

        Returns a (possibly empty) list of :class:`ValidationError`.
        Raises ``KeyError`` if *schema_name* is not registered.
        """
        schema = self._schemas[schema_name]
        errors: list[ValidationError] = []

        for f in schema.fields:
            if f.name not in data:
                if f.required:
                    errors.append(
                        ValidationError(
                            field=f.name,
                            message=f"Required field '{f.name}' is missing",
                            value=None,
                        )
                    )
                # Optional fields absent from data are fine
                continue

            value = data[f.name]
            if not _check_type(value, f.type_name):
                errors.append(
                    ValidationError(
                        field=f.name,
                        message=(
                            f"Field '{f.name}' expected type '{f.type_name}' "
                            f"but got {type(value).__name__}"
                        ),
                        value=value,
                    )
                )

        return errors

    def is_valid(self, data: dict, schema_name: str) -> bool:
        """Return ``True`` iff *data* has no validation errors."""
        return len(self.validate(data, schema_name)) == 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

PROTOCOL_REGISTRY: dict = {
    "schema_validator": SchemaValidator(),
}
