from __future__ import annotations

from ..unified_persona import PersonaFacet

DIALOGUE_FACET = PersonaFacet(
    facet_type="dialogue",
    config={
        "states": [
            "greeting",
            "information_gathering",
            "task_execution",
            "clarification",
            "confirmation",
            "closing",
        ]
    },
)
