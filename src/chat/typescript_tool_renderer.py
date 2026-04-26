"""TypeScript tool schema renderer for Aurelius.

Converts JSON-Schema tool definitions into TypeScript type declarations
suitable for injection into LLM prompts (e.g., GPT-OSS-style templates).

Usage::

    from src.chat.typescript_tool_renderer import TS_TOOL_RENDERER

    ts_block = TS_TOOL_RENDERER.render_namespace("functions", tools)
"""

from __future__ import annotations


def ts_type(
    schema: dict,
    required_fields: list[str] | None = None,
    indent: int = 0,
    field_name: str = "",
) -> str:
    """Recursively convert a JSON-Schema fragment to a TypeScript type string.

    Args:
        schema: A JSON-Schema dict (may be nested).
        required_fields: List of required field names in the *parent*
            object schema.  Used to decide whether to append ``| null``.
        indent: Current indentation depth (number of extra spaces = indent*2).
        field_name: The property name this schema belongs to (used for
            nullability check against *required_fields*).

    Returns:
        A TypeScript type string.
    """
    if required_fields is None:
        required_fields = []

    schema_type = schema.get("type", "")
    enum = schema.get("enum")

    # Enum wins over type.
    if enum is not None:
        ts = " | ".join(f'"{v}"' for v in enum)
        if field_name and field_name not in required_fields:
            ts += " | null"
        return ts

    indent_str = "  " * indent
    inner_indent = "  " * (indent + 1)

    if schema_type == "object":
        props = schema.get("properties", {})
        req = schema.get("required", [])
        if not props:
            ts = "Record<string, unknown>"
        else:
            lines: list[str] = ["{"]
            for k, v in props.items():
                opt = "" if k in req else "?"
                lines.append(f"{inner_indent}{k}{opt}: {ts_type(v, req, indent + 1, k)};")
            lines.append(f"{indent_str}}}")
            ts = "\n".join(lines)

    elif schema_type == "array":
        items = schema.get("items", {})
        item_ts = ts_type(items, [], indent, "")
        ts = f"{item_ts}[]"

    elif schema_type in ("integer", "number"):
        ts = "number"

    elif schema_type == "boolean":
        ts = "boolean"

    elif schema_type == "string":
        ts = "string"

    else:
        ts = "unknown"

    # Nullable if not in required list.
    if field_name and field_name not in required_fields:
        ts += " | null"

    return ts


def render_function_signature(tool: dict) -> str:
    """Render a single tool as a TypeScript ``type`` alias.

    Args:
        tool: A tool-schema dict with ``name``, ``description``, and
            ``parameters`` keys.

    Returns:
        A TypeScript type alias string, e.g.
        ``type my_func = (_: { arg1: string; }) => any;``
    """
    name = tool.get("name", "unknown")
    parameters = tool.get("parameters", {})
    required = parameters.get("required", [])
    props = parameters.get("properties", {})

    if not props:
        params_ts = "{}"
    else:
        prop_lines: list[str] = []
        for k, v in props.items():
            opt = "" if k in required else "?"
            prop_lines.append(f"    {k}{opt}: {ts_type(v, required, 2, k)};")
        params_ts = "{\n" + "\n".join(prop_lines) + "\n  }"

    return f"type {name} = (_: {params_ts}) => any;"


def render_namespace(namespace: str, tools: list[dict]) -> str:
    """Wrap rendered tool signatures in a TypeScript namespace block.

    Args:
        namespace: The namespace identifier (e.g. ``"functions"``).
        tools: List of tool-schema dicts.

    Returns:
        A complete TypeScript ``namespace { … }`` block as a string.
    """
    lines: list[str] = [f"## {namespace}", "", f"namespace {namespace} {{"]
    for tool in tools:
        description = tool.get("description", "")
        sig = render_function_signature(tool)
        if description:
            lines.append(f"  // {description}")
        lines.append(f"  {sig}")
    lines.append("}")
    return "\n".join(lines)


def render_tools_markdown(tools: list[dict]) -> str:
    """Render a list of tools as a Markdown-fenced TypeScript snippet.

    Args:
        tools: List of tool-schema dicts.

    Returns:
        A string beginning with ``## Functions`` followed by each
        tool's signature.
    """
    sigs = "\n".join(render_function_signature(t) for t in tools)
    return f"## Functions\n\n{sigs}"


class TypeScriptToolRenderer:
    """Thin class wrapper around the module-level rendering functions.

    Provides a convenient object interface (``TS_TOOL_RENDERER``) so
    other modules can do::

        from src.chat.typescript_tool_renderer import TS_TOOL_RENDERER
        block = TS_TOOL_RENDERER.render_namespace("functions", tools)
    """

    @staticmethod
    def ts_type(
        schema: dict,
        required_fields: list[str] | None = None,
        indent: int = 0,
        field_name: str = "",
    ) -> str:
        return ts_type(schema, required_fields, indent, field_name)

    @staticmethod
    def render_function_signature(tool: dict) -> str:
        return render_function_signature(tool)

    @staticmethod
    def render_namespace(namespace: str, tools: list[dict]) -> str:
        return render_namespace(namespace, tools)

    @staticmethod
    def render_tools_markdown(tools: list[dict]) -> str:
        return render_tools_markdown(tools)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

TS_TOOL_RENDERER = TypeScriptToolRenderer()

__all__ = [
    "ts_type",
    "render_function_signature",
    "render_namespace",
    "render_tools_markdown",
    "TypeScriptToolRenderer",
    "TS_TOOL_RENDERER",
]
