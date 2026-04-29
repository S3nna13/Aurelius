from __future__ import annotations

from ..unified_persona import PersonaFacet

CONSTITUTION_GENERAL_FACET = PersonaFacet(
    facet_type="constitution",
    config={"dimensions": ["level1.helpfulness", "level2.honesty", "level2.corrigibility"]},
)

CONSTITUTION_SECURITY_FACET = PersonaFacet(
    facet_type="constitution",
    config={"dimensions": ["level2.hard_constraints", "level2.harm_avoidance"]},
)

CONSTITUTION_AGENT_FACET = PersonaFacet(
    facet_type="constitution",
    config={"dimensions": ["level1.helpfulness", "level2.honesty"]},
)

HARM_FILTER_GENERAL_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": "all", "action": "block", "block_threshold": 0.5},
)

HARM_FILTER_SECURITY_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": ["malicious_code", "criminal_planning"], "action": "warn", "threshold": 0.8},
)

HARM_FILTER_CODING_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": ["malicious_code"], "action": "block"},
)

HARM_FILTER_THREAT_INTEL_FACET = PersonaFacet(
    facet_type="harm_filter",
    config={"categories": ["criminal_planning"], "action": "block"},
)
