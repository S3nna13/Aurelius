from __future__ import annotations

from ..unified_persona import PersonaFacet

AGENT_MODE_CODE_FACET = PersonaFacet(facet_type="agent_mode", config={"mode": "code"})
AGENT_MODE_ARCHITECT_FACET = PersonaFacet(facet_type="agent_mode", config={"mode": "architect"})
AGENT_MODE_ASK_FACET = PersonaFacet(facet_type="agent_mode", config={"mode": "ask"})
AGENT_MODE_DEBUG_FACET = PersonaFacet(facet_type="agent_mode", config={"mode": "debug"})
