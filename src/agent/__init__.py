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


# Register the ReAct loop under its canonical key. Import is deferred
# until after the registries exist so that cycles cannot form.
from .react_loop import AgentStep, AgentTrace, ReActLoop  # noqa: E402

AGENT_LOOP_REGISTRY["react"] = ReActLoop


__all__ = [
    "AGENT_LOOP_REGISTRY",
    "AgentStep",
    "AgentTrace",
    "ReActLoop",
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


# --- safe tool dispatcher (additive) ----------------------------------------
from .tool_registry_dispatcher import (  # noqa: E402
    SessionBudget,
    ToolInvocationResult,
    ToolRegistryDispatcher,
    ToolSpec,
)

AGENT_LOOP_REGISTRY["safe_dispatch"] = ToolRegistryDispatcher

__all__ += [
    "SessionBudget",
    "ToolInvocationResult",
    "ToolRegistryDispatcher",
    "ToolSpec",
]


# --- tree-of-thoughts beam planner (additive) -------------------------------
from .agent_planner import BeamPlanner, PlanNode  # noqa: E402

AGENT_LOOP_REGISTRY["beam_plan"] = BeamPlanner

__all__ += [
    "BeamPlanner",
    "PlanNode",
]


# --- tool error recovery (additive) -----------------------------------------
from .tool_error_recovery import (  # noqa: E402
    RecoveringDispatcher,
    RecoveryDecision,
    RecoveryPolicy,
    RecoveryStrategy,
)

__all__ += [
    "RecoveringDispatcher",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RecoveryStrategy",
]
