"""Persona facets — composable capability attachments."""

from .security_facet import create_security_facet, render_security_facet_config
from .threat_intel_facet import create_threat_intel_facet, classify_query, validate_threat_intel_response
from .agent_mode_facet import create_agent_mode_facet, is_tool_allowed, get_allowed_tools
from .constitution_facet import create_constitution_facet, get_active_dimensions, score_response
from .harm_filter_facet import create_harm_filter_facet, classify_harm, HARM_CATEGORIES
from .personality_facet import create_personality_facet, score_input
from .dialogue_facet import create_dialogue_facet, classify_transition, DialogueState

__all__ = [
    "create_security_facet",
    "render_security_facet_config",
    "create_threat_intel_facet",
    "classify_query",
    "validate_threat_intel_response",
    "create_agent_mode_facet",
    "is_tool_allowed",
    "get_allowed_tools",
    "create_constitution_facet",
    "get_active_dimensions",
    "score_response",
    "create_harm_filter_facet",
    "classify_harm",
    "HARM_CATEGORIES",
    "create_personality_facet",
    "score_input",
    "create_dialogue_facet",
    "classify_transition",
    "DialogueState",
]