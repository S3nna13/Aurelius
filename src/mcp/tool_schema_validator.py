"""Aurelius MCP tool schema validator.

Provides structural validation for MCP tool schemas: name format, description
presence, parameter shape, type constraints, circular ``$ref`` detection, and
schema size limits.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_ALLOWED_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

_MAX_SCHEMA_KEYS = 10_000


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class ToolSchemaValidator:
    """Validates MCP tool schemas for structural correctness.

    Checks:
    - name is non-empty alphanumeric + underscore
    - description is present and non-empty
    - parameters is a dict with 'properties' and 'required' keys
    - parameter types are in {"string", "integer", "number", "boolean", "array", "object"}
    - no circular $ref references
    """

    def validate(self, schema: dict[str, Any]) -> list[str]:
        """Validate *schema* and return a list of human-readable error messages.

        An empty list means the schema is valid.
        """
        errors: list[str] = []

        if not isinstance(schema, dict):
            errors.append("Schema must be a dict")
            return errors

        # Name checks
        name = schema.get("name")
        if name is None:
            errors.append("Missing required field: 'name'")
        elif not isinstance(name, str):
            errors.append(f"'name' must be a string, got {type(name).__name__}")
        elif not name.strip():
            errors.append("'name' must be a non-empty string")
        elif not _NAME_RE.match(name):
            errors.append(
                f"'name' must match '^[a-zA-Z_][a-zA-Z0-9_]*$', got {name!r}"
            )

        # Description checks
        description = schema.get("description")
        if description is None:
            errors.append("Missing required field: 'description'")
        elif not isinstance(description, str):
            errors.append(
                f"'description' must be a string, got {type(description).__name__}"
            )
        elif not description.strip():
            errors.append("'description' must be a non-empty string")

        # Circular $ref check (run before size check to avoid recursion)
        ref_error = _detect_circular_ref(schema)
        if ref_error:
            errors.append(ref_error)

        # Size check
        key_count = _count_keys(schema)
        if key_count > _MAX_SCHEMA_KEYS:
            errors.append(
                f"Schema size ({key_count} keys) exceeds maximum allowed "
                f"({_MAX_SCHEMA_KEYS} keys)"
            )

        # Parameters checks
        parameters = schema.get("parameters")
        if parameters is None:
            errors.append("Missing required field: 'parameters'")
        elif not isinstance(parameters, dict):
            errors.append(
                f"'parameters' must be a dict, got {type(parameters).__name__}"
            )
        else:
            if "properties" not in parameters:
                errors.append("'parameters' missing required key: 'properties'")
            elif not isinstance(parameters["properties"], dict):
                errors.append(
                    "'parameters.properties' must be a dict, got "
                    f"{type(parameters['properties']).__name__}"
                )
            else:
                for prop_name, prop_def in parameters["properties"].items():
                    if not isinstance(prop_def, dict):
                        errors.append(
                            f"Property {prop_name!r} must be a dict, got "
                            f"{type(prop_def).__name__}"
                        )
                        continue
                    prop_type = prop_def.get("type")
                    if prop_type is None:
                        errors.append(
                            f"Property {prop_name!r} missing required key: 'type'"
                        )
                    elif prop_type not in _ALLOWED_TYPES:
                        errors.append(
                            f"Property {prop_name!r} has invalid type "
                            f"{prop_type!r}; allowed types are "
                            f"{sorted(_ALLOWED_TYPES)!r}"
                        )

            if "required" not in parameters:
                errors.append("'parameters' missing required key: 'required'")
            elif not isinstance(parameters["required"], list):
                errors.append(
                    "'parameters.required' must be a list, got "
                    f"{type(parameters['required']).__name__}"
                )

        return errors

    def is_valid(self, schema: dict[str, Any]) -> bool:
        """Return ``True`` if *schema* passes all validation checks."""
        return len(self.validate(schema)) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_keys(obj: Any) -> int:
    """Recursively count the total number of dict keys in *obj*."""
    seen: set[int] = set()

    def _walk(node: Any) -> int:
        if not isinstance(node, dict):
            return 0
        node_id = id(node)
        if node_id in seen:
            return 0
        seen.add(node_id)
        count = len(node)
        for value in node.values():
            if isinstance(value, dict):
                count += _walk(value)
            elif isinstance(value, list):
                for item in value:
                    count += _walk(item)
        return count

    return _walk(obj)


def _detect_circular_ref(obj: Any) -> str | None:
    """Return an error string if a circular ``$ref`` is found, else ``None``.

    A cycle is reported only when at least one node in the cycle contains a
    ``$ref`` key, matching the JSON Schema reference semantics.
    """
    path_ids: set[int] = set()
    has_ref_in_path: dict[int, bool] = {}

    def _walk(node: Any, path: str, parent_has_ref: bool) -> str | None:
        if isinstance(node, dict):
            node_id = id(node)
            current_has_ref = parent_has_ref or "$ref" in node
            if node_id in path_ids:
                if current_has_ref:
                    return f"Circular $ref detected at {path}"
                return None
            path_ids.add(node_id)
            has_ref_in_path[node_id] = current_has_ref
            for key, value in node.items():
                result = _walk(
                    value, f"{path}.{key}", has_ref_in_path.get(node_id, False)
                )
                if result:
                    return result
            path_ids.discard(node_id)
            has_ref_in_path.pop(node_id, None)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                result = _walk(item, f"{path}[{idx}]", parent_has_ref)
                if result:
                    return result
        return None

    return _walk(obj, "schema", False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULT_TOOL_SCHEMA_VALIDATOR = ToolSchemaValidator()

#: Maps logical validator names to ``ToolSchemaValidator`` instances.
TOOL_SCHEMA_VALIDATOR_REGISTRY: dict[str, ToolSchemaValidator] = {
    "default": DEFAULT_TOOL_SCHEMA_VALIDATOR,
}

__all__ = [
    "DEFAULT_TOOL_SCHEMA_VALIDATOR",
    "TOOL_SCHEMA_VALIDATOR_REGISTRY",
    "ToolSchemaValidator",
]
