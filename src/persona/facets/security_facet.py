"""Security facet — enforces scope, guardrails, and workflow stages for security personas."""

from __future__ import annotations

from ..unified_persona import PersonaFacet


def create_security_facet(mode: str = "defensive", scope: str = "all") -> PersonaFacet:
    return PersonaFacet(
        facet_type="security",
        config={"mode": mode, "scope": scope},
    )


def render_security_facet_config(facet: PersonaFacet) -> dict:
    cfg = facet.config
    return {
        "mode": cfg.get("mode", "defensive"),
        "scope": cfg.get("scope", "all"),
        "enforces_output_contract": True,
        "enforces_workflow_stages": True,
        "requires_scope_declaration": cfg.get("mode") == "offensive",
    }