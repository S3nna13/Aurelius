from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParameterSchema:
    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any = None
    enum: list | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ParameterSchema]
    returns: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)


class ToolSchemaRegistry:
    """Central registry of all available tool schemas."""

    def __init__(self) -> None:
        self._schemas: dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema) -> None:
        self._schemas[schema.name] = schema

    def get(self, name: str) -> ToolSchema | None:
        return self._schemas.get(name)

    def list_tools(self) -> list[str]:
        return list(self._schemas.keys())

    def list_by_tag(self, tag: str) -> list[ToolSchema]:
        return [s for s in self._schemas.values() if tag in s.tags]

    def unregister(self, name: str) -> bool:
        if name in self._schemas:
            del self._schemas[name]
            return True
        return False

    def to_dict(self, name: str) -> dict | None:
        schema = self._schemas.get(name)
        if schema is None:
            return None
        return self._schema_to_dict(schema)

    def all_to_dict(self) -> list[dict]:
        return [self._schema_to_dict(s) for s in self._schemas.values()]

    def _schema_to_dict(self, schema: ToolSchema) -> dict:
        params = []
        for p in schema.parameters:
            entry: dict[str, Any] = {
                "name": p.name,
                "type": p.type,
                "required": p.required,
                "description": p.description,
                "default": p.default,
            }
            if p.enum is not None:
                entry["enum"] = list(p.enum)
            params.append(entry)
        return {
            "name": schema.name,
            "description": schema.description,
            "parameters": params,
            "returns": schema.returns,
            "version": schema.version,
            "tags": list(schema.tags),
        }


TOOL_SCHEMA_REGISTRY = ToolSchemaRegistry()
