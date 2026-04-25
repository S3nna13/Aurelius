"""Input schema validator for agent tool call parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_type_hints


@dataclass
class ParamSpec:
    name: str
    param_type: type
    required: bool = True
    default: Any = None
    description: str = ""

    def validate(self, value: Any) -> tuple[bool, str]:
        if value is None:
            if self.required:
                return False, f"required parameter '{self.name}' is missing"
            return True, "ok"
        if not isinstance(value, self.param_type):
            return False, f"parameter '{self.name}' expected {self.param_type.__name__}, got {type(value).__name__}"
        return True, "ok"


@dataclass
class InputSchema:
    params: list[ParamSpec] = field(default_factory=list)

    def add(self, spec: ParamSpec) -> None:
        self.params.append(spec)

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        errors = []
        for spec in self.params:
            val = data.get(spec.name)
            ok, msg = spec.validate(val)
            if not ok:
                errors.append(msg)
        return len(errors) == 0, errors


INPUT_SCHEMA = InputSchema()