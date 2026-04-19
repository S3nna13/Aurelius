"""Aurelius agent surface: tool-call parsing and agent-loop registry.

This module is deliberately minimal: it exposes two registries other
surfaces populate and registers this surface's own tool-call parsers.
No heavyweight imports; no side effects beyond registry population.
"""

from __future__ import annotations

from typing import Callable

from .tool_call_parser import (
    JSONToolCallParser,
    ParsedToolCall,
    ParseResult,
    ToolCallParseError,
    UnifiedToolCallParser,
    XMLToolCallParser,
    detect_format,
    format_json,
    format_xml,
    parse_json,
    parse_xml,
)

#: Registry of named tool-call parser callables (``text -> list[ParsedToolCall]``).
#: Keys are format identifiers ("xml", "json"); additional siblings may
#: register more at import time.
TOOL_CALL_PARSER_REGISTRY: dict[str, Callable[[str], list[ParsedToolCall]]] = {}

#: Registry for agent-loop implementations. Populated by downstream
#: modules (react loop, plan-and-execute, etc.); left empty here.
AGENT_LOOP_REGISTRY: dict[str, Callable[..., object]] = {}


# Register this surface's parsers under their canonical keys.
TOOL_CALL_PARSER_REGISTRY["xml"] = parse_xml
TOOL_CALL_PARSER_REGISTRY["json"] = parse_json


__all__ = [
    "AGENT_LOOP_REGISTRY",
    "JSONToolCallParser",
    "ParsedToolCall",
    "ParseResult",
    "TOOL_CALL_PARSER_REGISTRY",
    "ToolCallParseError",
    "UnifiedToolCallParser",
    "XMLToolCallParser",
    "detect_format",
    "format_json",
    "format_xml",
    "parse_json",
    "parse_xml",
]
