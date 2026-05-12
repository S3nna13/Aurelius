"""Function-calling API shape validator for Aurelius serving.

OpenAI/Claude-compatible function-calling schema-only validation layer.
Validates incoming ``tools`` + ``tool_choice`` request shapes, parses
model-emitted ``tool_calls``, and formats ``tool`` role messages.

This module does NOT execute tools — execution lives in
``src/serving/tool_executor.py``. This is a pure wire-shape contract.

Pure stdlib only: ``dataclasses``, ``json``, ``typing``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

__all__ = [
    "FunctionSchema",
    "ToolDefinition",
    "ToolCall",
    "ToolChoice",
    "FunctionCallError",
    "FunctionCallValidator",
    "DEFAULT_TOOL_CHOICE",
    "ALLOWED_TYPES",
]


ALLOWED_TYPES: tuple[str, ...] = ("function",)
_ALLOWED_TOOL_CHOICE_MODES = frozenset({"auto", "none", "required", "named"})


class FunctionCallError(Exception):
    """Raised on any schema / wire-shape violation in the function-calling API."""


@dataclass(frozen=True)
class FunctionSchema:
    """JSON-Schema-style function description.

    ``parameters`` is expected to be a JSON-Schema object with at minimum
    ``{"type": "object", "properties": {...}, "required": [...]}``.
    """

    name: str
    description: str
    parameters: dict


@dataclass
class ToolDefinition:
    """A single tool offered to the model. Only ``type == "function"``."""

    function: FunctionSchema
    type: str = "function"


@dataclass
class ToolCall:
    """A model-emitted tool call. ``arguments_json`` must parse as JSON."""

    id: str
    type: str
    function_name: str
    arguments_json: str


@dataclass
class ToolChoice:
    """User-supplied tool_choice directive.

    mode ∈ {"auto", "none", "required", "named"}. "named" requires ``name``.
    """

    mode: str
    name: str | None = None


DEFAULT_TOOL_CHOICE: ToolChoice = ToolChoice(mode="auto")


class FunctionCallValidator:
    """Stateless validator for function-calling API shapes.

    All ``validate_*`` methods raise ``FunctionCallError`` on failure and
    return ``None`` on success. Parse methods return concrete dataclasses.
    """

    # -- Function schema ---------------------------------------------------

    def validate_function_schema_json_compatible(self, schema: dict) -> None:
        """Ensure a JSON-Schema-style dict is shaped for OpenAI tool use.

        Requires ``type == "object"``, a ``properties`` dict, and a
        ``required`` list whose entries all appear in ``properties``.
        Properties without an explicit ``type`` are accepted (permissive
        policy — matches OpenAI/Claude leniency where absent ``type``
        means any-JSON-value).
        """
        if not isinstance(schema, dict):
            raise FunctionCallError(
                f"parameters schema must be a dict, got {type(schema).__name__}"
            )
        if schema.get("type") != "object":
            raise FunctionCallError("parameters schema must have type == 'object'")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise FunctionCallError("parameters schema must include a 'properties' dict")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise FunctionCallError("parameters schema 'required' must be a list")
        for rname in required:
            if not isinstance(rname, str):
                raise FunctionCallError(f"required entry must be str, got {type(rname).__name__}")
            if rname not in properties:
                raise FunctionCallError(f"required field {rname!r} missing from properties")
        # Permissive on property-level type: absent == accept. Present but
        # non-string is a structural error.
        for pname, pspec in properties.items():
            if not isinstance(pname, str):
                raise FunctionCallError(f"property name must be str, got {type(pname).__name__}")
            if not isinstance(pspec, dict):
                raise FunctionCallError(f"property {pname!r} spec must be a dict")
            if "type" in pspec and not isinstance(pspec["type"], (str, list)):
                raise FunctionCallError(f"property {pname!r} 'type' must be str or list")

    # -- Tool definition ---------------------------------------------------

    def validate_tool_definition(self, td: ToolDefinition) -> None:
        if not isinstance(td, ToolDefinition):
            raise FunctionCallError(f"expected ToolDefinition, got {type(td).__name__}")
        if td.type not in ALLOWED_TYPES:
            raise FunctionCallError(f"tool type must be one of {ALLOWED_TYPES}, got {td.type!r}")
        fn = td.function
        if not isinstance(fn, FunctionSchema):
            raise FunctionCallError("ToolDefinition.function must be a FunctionSchema")
        if not isinstance(fn.name, str) or not fn.name:
            raise FunctionCallError("function name must be a non-empty str")
        if not isinstance(fn.description, str):
            raise FunctionCallError("function description must be a str")
        self.validate_function_schema_json_compatible(fn.parameters)

    # -- Tool choice -------------------------------------------------------

    def validate_tool_choice(self, tc: ToolChoice, available: list[str]) -> None:
        if not isinstance(tc, ToolChoice):
            raise FunctionCallError(f"expected ToolChoice, got {type(tc).__name__}")
        if tc.mode not in _ALLOWED_TOOL_CHOICE_MODES:
            raise FunctionCallError(
                f"tool_choice.mode must be one of "
                f"{sorted(_ALLOWED_TOOL_CHOICE_MODES)}, got {tc.mode!r}"
            )
        if tc.mode == "named":
            if tc.name is None:
                raise FunctionCallError("tool_choice mode='named' requires a 'name'")
            if tc.name not in available:
                raise FunctionCallError(
                    f"tool_choice name {tc.name!r} not in available tools {available!r}"
                )
        else:
            # name must not be set for non-named modes (strict contract)
            if tc.name is not None:
                raise FunctionCallError(f"tool_choice.name must be None for mode={tc.mode!r}")

    # -- Tool calls --------------------------------------------------------

    def parse_tool_calls(self, raw: list[dict]) -> list[ToolCall]:
        if not isinstance(raw, list):
            raise FunctionCallError(f"tool_calls must be a list, got {type(raw).__name__}")
        parsed: list[ToolCall] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise FunctionCallError(
                    f"tool_calls[{i}] must be a dict, got {type(entry).__name__}"
                )
            call_id = entry.get("id")
            call_type = entry.get("type")
            function = entry.get("function")
            if not isinstance(call_id, str) or not call_id:
                raise FunctionCallError(f"tool_calls[{i}].id must be a non-empty str")
            if call_type not in ALLOWED_TYPES:
                raise FunctionCallError(
                    f"tool_calls[{i}].type must be one of {ALLOWED_TYPES}, got {call_type!r}"
                )
            if not isinstance(function, dict):
                raise FunctionCallError(f"tool_calls[{i}].function must be a dict")
            fname = function.get("name")
            args = function.get("arguments")
            if not isinstance(fname, str) or not fname:
                raise FunctionCallError(f"tool_calls[{i}].function.name must be a non-empty str")
            if not isinstance(args, str):
                raise FunctionCallError(
                    f"tool_calls[{i}].function.arguments must be a JSON "
                    f"string, got {type(args).__name__}"
                )
            try:
                json.loads(args)
            except json.JSONDecodeError as exc:
                raise FunctionCallError(
                    f"tool_calls[{i}].function.arguments is not valid JSON: {exc.msg}"
                ) from exc
            parsed.append(
                ToolCall(
                    id=call_id,
                    type=call_type,
                    function_name=fname,
                    arguments_json=args,
                )
            )
        return parsed

    # -- Tool-role message formatting -------------------------------------

    def format_tool_message(self, tool_call_id: str, name: str, content: str) -> dict:
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise FunctionCallError("tool_call_id must be a non-empty str")
        if not isinstance(name, str) or not name:
            raise FunctionCallError("name must be a non-empty str")
        if not isinstance(content, str):
            raise FunctionCallError("content must be a str")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }
