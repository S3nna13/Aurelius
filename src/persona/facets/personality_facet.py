from __future__ import annotations

from ..unified_persona import PersonaFacet

PERSONALITY_PATIENT_FACET = PersonaFacet(
    facet_type="personality",
    config={"traits": ["patient", "encouraging", "socratic"]},
)

PERSONALITY_ANALYTICAL_FACET = PersonaFacet(
    facet_type="personality",
    config={"traits": ["analytical", "precise", "evidence-first"]},
)

PERSONALITY_CREATIVE_FACET = PersonaFacet(
    facet_type="personality",
    config={"traits": ["imaginative", "expressive"]},
)

PERSONALITY_THOROUGH_FACET = PersonaFacet(
    facet_type="personality",
    config={"traits": ["thorough", "patient"]},
)

PERSONALITY_METHODICAL_FACET = PersonaFacet(
    facet_type="personality",
    config={"traits": ["methodical", "investigative"]},
)
