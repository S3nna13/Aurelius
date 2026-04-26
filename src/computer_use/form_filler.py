"""
form_filler.py
Automates form field filling with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FieldType(Enum):
    TEXT = "text"
    PASSWORD = "password"
    EMAIL = "email"
    NUMBER = "number"
    CHECKBOX = "checkbox"
    SELECT = "select"
    TEXTAREA = "textarea"


@dataclass
class FormField:
    field_id: str
    field_type: FieldType
    label: str = ""
    placeholder: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FormFillResult:
    field_id: str
    value: str
    success: bool
    error: str = ""


class FormFiller:
    def __init__(self) -> None:
        self._fields: dict[str, FormField] = {}

    def register_field(self, field: FormField) -> None:
        self._fields[field.field_id] = field

    def fill(self, field_id: str, value: str) -> FormFillResult:
        if field_id not in self._fields:
            return FormFillResult(
                field_id=field_id,
                value=value,
                success=False,
                error=f"Field '{field_id}' not registered.",
            )

        f = self._fields[field_id]

        # Required check (before type-specific validation)
        if f.required and value == "":
            return FormFillResult(
                field_id=field_id,
                value=value,
                success=False,
                error=f"Field '{field_id}' is required.",
            )

        # Type-specific validation
        if f.field_type == FieldType.EMAIL:
            if "@" not in value:
                return FormFillResult(
                    field_id=field_id,
                    value=value,
                    success=False,
                    error="Email must contain '@'.",
                )

        elif f.field_type == FieldType.NUMBER:
            try:
                float(value)
            except ValueError:
                return FormFillResult(
                    field_id=field_id,
                    value=value,
                    success=False,
                    error=f"Value '{value}' is not a valid number.",
                )

        elif f.field_type == FieldType.CHECKBOX:
            if value not in ("true", "false"):
                return FormFillResult(
                    field_id=field_id,
                    value=value,
                    success=False,
                    error="Checkbox value must be 'true' or 'false'.",
                )

        elif f.field_type == FieldType.SELECT:
            if value not in f.options:
                return FormFillResult(
                    field_id=field_id,
                    value=value,
                    success=False,
                    error=f"Value '{value}' is not a valid option. Choices: {f.options}",
                )

        return FormFillResult(field_id=field_id, value=value, success=True)

    def fill_form(self, values: dict[str, str]) -> list[FormFillResult]:
        return [self.fill(field_id, value) for field_id, value in values.items()]

    def validate(self, field_id: str, value: str) -> bool:
        return self.fill(field_id, value).success


FORM_FILLER_REGISTRY: dict[str, type] = {"default": FormFiller}

REGISTRY = FORM_FILLER_REGISTRY
