"""Aurelius MCP tool registry with schema validation.

Provides dataclasses and a registry class for declaring, validating, and
looking up MCP tools, together with OpenAI-compatible schema export.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolParameter:
    """Describes a single parameter accepted by an MCP tool.

    Attributes:
        name:        Parameter identifier.
        type:        JSON Schema primitive type (e.g. "string", "integer").
        required:    Whether the parameter must be supplied by callers.
        description: Human-readable explanation of the parameter's purpose.
    """

    name: str
    type: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ToolDefinition:
    """Declarative manifest for a single MCP tool.

    Attributes:
        name:        Unique tool identifier.
        description: Human-readable description of what the tool does.
        parameters:  Ordered list of :class:`ToolParameter` descriptors.
        version:     Semantic version string (default ``"1.0.0"``).
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry for MCP tool definitions with call-site validation.

    Usage::

        registry = ToolRegistry()
        tool = ToolDefinition(
            name="search",
            description="Web search",
            parameters=[ToolParameter("query", "string")],
        )
        registry.register(tool)
        missing = registry.validate_call("search", {"query": "hello"})
        # missing == []
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        """Register *tool* in the catalog.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool {tool.name!r} is already registered. "
                "Unregister it first if you want to replace it."
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove the tool registered under *name*.

        Returns:
            ``True`` if the tool was present and removed, ``False`` otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition | None:
        """Return the :class:`ToolDefinition` for *name*, or ``None``."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_call(self, name: str, kwargs: dict) -> list[str]:
        """Validate a prospective tool call against the registered schema.

        Parameters
        ----------
        name:
            Tool name to look up.
        kwargs:
            Argument dict supplied by the caller.

        Returns
        -------
        list[str]
            Empty list when the call is valid; a list of missing required
            parameter names otherwise.  Returns ``["tool_not_found"]`` if
            no tool is registered under *name*.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ["tool_not_found"]
        return [
            p.name
            for p in tool.parameters
            if p.required and p.name not in kwargs
        ]

    # ------------------------------------------------------------------
    # Schema export
    # ------------------------------------------------------------------

    def to_schema(self) -> list[dict]:
        """Export all registered tools as an OpenAI-compatible schema list.

        Each entry has the form::

            {
                "name": "<tool-name>",
                "description": "<description>",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "<param-name>": {
                            "type": "<param-type>",
                            "description": "<param-description>",
                        },
                        ...
                    },
                    "required": ["<required-param-name>", ...],
                },
            }
        """
        schemas: list[dict] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            properties: dict[str, dict] = {
                p.name: {"type": p.type, "description": p.description}
                for p in tool.parameters
            }
            required_names: list[str] = [
                p.name for p in tool.parameters if p.required
            ]
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_names,
                    },
                }
            )
        return schemas


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps logical registry names to :class:`ToolRegistry` classes.
TOOL_REGISTRY_REGISTRY: dict[str, type[ToolRegistry]] = {"default": ToolRegistry}

__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "TOOL_REGISTRY_REGISTRY",
]
