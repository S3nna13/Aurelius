from __future__ import annotations

from ..unified_persona import Guardrail, PersonaFacet

SECURITY_OFFENSIVE_FACET = PersonaFacet(
    facet_type="security",
    config={"mode": "offensive", "scope": "closed_internal"},
)

SECURITY_DEFENSIVE_FACET = PersonaFacet(
    facet_type="security",
    config={"mode": "defensive"},
)

SECURITY_PURPLE_FACET = PersonaFacet(
    facet_type="security",
    config={"mode": "purple", "scope": "closed_internal"},
)


class SecurityFacet:
    @staticmethod
    def make_guardrails(mode: str) -> list[Guardrail]:
        base = [
            Guardrail("cite_sources", "Prioritize verified observables. Never fabricate evidence.", "high"),
            Guardrail("calibrated_uncertainty", "If detail is disputed, say so with confidence qualifier.", "medium"),
        ]
        if mode == "offensive":
            base.append(Guardrail("no_exploit_code", "Never provide working exploit code for third-party production systems.", "critical"))
            base.append(Guardrail("scope_boundary", "Operate only on authorized in-scope assets.", "critical"))
        return base
