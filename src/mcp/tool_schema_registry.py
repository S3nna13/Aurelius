"""Aurelius MCP tool manifest and schema registry.

Provides dataclasses and registry helpers for declaring, validating, and
looking up tool schemas used across the MCP surface.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolSchemaError(Exception):
    """Raised for invalid tool schemas, version mismatches, or failed validation."""


# ---------------------------------------------------------------------------
# ToolSchema dataclass
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")


@dataclass
class ToolSchema:
    """Declarative manifest for a single MCP tool.

    Attributes:
        name: Unique identifier for the tool (non-empty).
        description: Human-readable description of what the tool does.
        input_schema: JSON Schema dict describing the expected input.
        output_schema: JSON Schema dict describing the expected output.
        version: Semantic version string (``MAJOR.MINOR[.PATCH]``).
        tags: Arbitrary string labels for filtering / discovery.
    """

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolSchemaError("ToolSchema.name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ToolSchemaError("ToolSchema.description must be a string")
        if not isinstance(self.input_schema, dict):
            raise ToolSchemaError("ToolSchema.input_schema must be a dict")
        if not isinstance(self.output_schema, dict):
            raise ToolSchemaError("ToolSchema.output_schema must be a dict")
        if not isinstance(self.version, str) or not _VERSION_RE.match(self.version):
            raise ToolSchemaError(
                f"ToolSchema.version must match MAJOR.MINOR[.PATCH], got {self.version!r}"
            )
        if not isinstance(self.tags, list):
            raise ToolSchemaError("ToolSchema.tags must be a list")

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def major_version(self) -> int:
        """Return the major version component as an integer."""
        return int(self.version.split(".")[0])

    def is_compatible_with(self, other: "ToolSchema") -> bool:
        """Return True if *other* shares the same major version as *self*.

        Compatibility follows semver: differing major versions are breaking.
        """
        return self.major_version() == other.major_version()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps tool names to their ``ToolSchema`` instances.
MCP_TOOL_SCHEMA_REGISTRY: dict[str, ToolSchema] = {}


def register_tool_schema(schema: ToolSchema) -> None:
    """Register *schema* in ``MCP_TOOL_SCHEMA_REGISTRY``.

    If a schema with the same name already exists and has an incompatible
    major version, raises ``ToolSchemaError``.  Compatible re-registrations
    (same major version, different minor/patch) silently update the entry.

    Args:
        schema: A fully-constructed ``ToolSchema`` instance.

    Raises:
        ToolSchemaError: If *schema* is not a ``ToolSchema``, has a missing
            name, or has an incompatible version with a previously registered
            schema of the same name.
    """
    if not isinstance(schema, ToolSchema):
        raise ToolSchemaError(
            f"register_tool_schema expects a ToolSchema instance, got {type(schema)!r}"
        )
    existing = MCP_TOOL_SCHEMA_REGISTRY.get(schema.name)
    if existing is not None and not existing.is_compatible_with(schema):
        raise ToolSchemaError(
            f"Version mismatch for tool {schema.name!r}: "
            f"registered version {existing.version!r} (major {existing.major_version()}) "
            f"is incompatible with new version {schema.version!r} (major {schema.major_version()})"
        )
    MCP_TOOL_SCHEMA_REGISTRY[schema.name] = schema


def get_tool_schema(name: str) -> ToolSchema:
    """Return the ``ToolSchema`` registered under *name*.

    Raises:
        KeyError: If no schema is registered for *name*.
    """
    if name not in MCP_TOOL_SCHEMA_REGISTRY:
        raise KeyError(
            f"No tool schema registered for {name!r}. "
            f"Available: {sorted(MCP_TOOL_SCHEMA_REGISTRY)!r}"
        )
    return MCP_TOOL_SCHEMA_REGISTRY[name]


def validate_tool_call(name: str, args: dict) -> bool:
    """Validate *args* against the registered schema for tool *name*.

    Performs a structural check: all ``required`` keys listed in
    ``input_schema`` must be present in *args*.

    Args:
        name: The tool name to look up.
        args: The argument dict to validate.

    Returns:
        ``True`` if validation passes.

    Raises:
        KeyError: If no schema is registered for *name*.
        ToolSchemaError: If required keys are absent from *args*.
    """
    schema = get_tool_schema(name)
    required = schema.input_schema.get("required", [])
    if not isinstance(required, list):
        raise ToolSchemaError(
            f"input_schema 'required' for {name!r} must be a list, "
            f"got {type(required).__name__}"
        )
    missing = [k for k in required if k not in args]
    if missing:
        raise ToolSchemaError(
            f"Tool {name!r} call missing required argument(s): {missing!r}"
        )
    return True


__all__ = [
    "MCP_TOOL_SCHEMA_REGISTRY",
    "ToolSchema",
    "ToolSchemaError",
    "get_tool_schema",
    "register_tool_schema",
    "validate_tool_call",
]
