from __future__ import annotations

from ..unified_persona import PersonaFacet

HARM_FILTER_ALL_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": "all", "block_threshold": 0.5, "action": "block"},
)

HARM_FILTER_WARN_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": "all", "block_threshold": 0.8, "action": "warn"},
)
