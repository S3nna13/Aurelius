"""Agent mode facet — tool gating and response style for agent personas."""

from __future__ import annotations

from ..unified_persona import PersonaFacet


def create_agent_mode_facet(mode: str = "code", allowed_tools: tuple[str, ...] = ()) -> PersonaFacet:
    return PersonaFacet(
        facet_type="agent_mode",
        config={"mode": mode, "allowed_tools": list(allowed_tools)},
    )


def is_tool_allowed(facet: PersonaFacet, tool_name: str) -> bool:
    tools = facet.config.get("allowed_tools", [])
    if not tools:
        return True
    return tool_name in tools


def get_allowed_tools(facet: PersonaFacet) -> list[str]:
    return facet.config.get("allowed_tools", [])