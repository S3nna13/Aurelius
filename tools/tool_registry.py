"""Core tool registry: ToolSpec definitions, invocation dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    required: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, spec: ToolSpec, handler: Callable) -> None:
        """Store spec + handler by name."""
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def invoke(self, name: str, **kwargs) -> ToolResult:
        """Call handler(**kwargs), wraps output in ToolResult; catches Exception."""
        if name not in self._handlers:
            return ToolResult(
                tool_name=name, success=False, output="", error=f"unknown tool: {name!r}"
            )
        try:
            result = self._handlers[name](**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(tool_name=name, success=True, output=str(result))
        except Exception as e:
            return ToolResult(tool_name=name, success=False, output="", error=str(e))

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def list_tools(self) -> list[str]:
        return list(self._specs.keys())

    def to_openai_format(self) -> list[dict]:
        """Returns OpenAI-compatible function tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs.values()
        ]


TOOL_REGISTRY = ToolRegistry()
