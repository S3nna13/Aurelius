"""Threat intelligence output contracts — CVE, MITRE, actor, and IOC JSON schemas.

Extracted from src/chat/threat_intel_persona.py to serve the unified persona system.
"""

from __future__ import annotations

from typing import Any

from ..unified_persona import OutputContract

CVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "cve_id",
        "affected_systems",
        "cvss_score",
        "cvss_vector",
        "exploit_status",
        "remediation",
        "references",
    ],
    "properties": {
        "cve_id": {"type": "string"},
        "affected_systems": {"type": "array", "items": {"type": "string"}},
        "cvss_score": {"type": "number"},
        "cvss_vector": {"type": "string"},
        "exploit_status": {"type": "string"},
        "remediation": {"type": "string"},
        "references": {"type": "array", "items": {"type": "string"}},
    },
}

MITRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "technique_id",
        "tactic",
        "description",
        "usage_examples",
        "detection",
        "mitigation",
        "sub_techniques",
    ],
    "properties": {
        "technique_id": {"type": "string"},
        "tactic": {"type": "string"},
        "description": {"type": "string"},
        "usage_examples": {"type": "array", "items": {"type": "string"}},
        "detection": {"type": "string"},
        "mitigation": {"type": "string"},
        "sub_techniques": {"type": "array", "items": {"type": "string"}},
    },
}

ACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "actor_name",
        "aliases",
        "ttps",
        "target_sectors",
        "notable_campaigns",
        "attribution_confidence",
    ],
    "properties": {
        "actor_name": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "ttps": {"type": "array", "items": {"type": "string"}},
        "target_sectors": {"type": "array", "items": {"type": "string"}},
        "notable_campaigns": {"type": "array", "items": {"type": "string"}},
        "attribution_confidence": {"type": "string"},
    },
}

IOC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "ioc_type",
        "value",
        "first_seen",
        "last_seen",
        "associated_actor",
        "confidence",
        "source_refs",
    ],
    "properties": {
        "ioc_type": {"type": "string"},
        "value": {"type": "string"},
        "first_seen": {"type": "string"},
        "last_seen": {"type": "string"},
        "associated_actor": {"type": "string"},
        "confidence": {"type": "string"},
        "source_refs": {"type": "array", "items": {"type": "string"}},
    },
}

THREAT_INTEL_GUARDRAILS = (
    "Never provide working exploit code in this persona. Describe vulnerability class and mitigations only.",
    "Prefer primary sources: NVD, MITRE CVE / ATT&CK, vendor advisories, CISA KEV.",
    "Maintain calibrated uncertainty. If attribution or detail is disputed, say so with a confidence qualifier.",
)

CVE_CONTRACT = OutputContract(
    name="cve",
    schema=CVE_SCHEMA,
    required_fields=("cve_id", "affected_systems", "cvss_score", "remediation"),
)

MITRE_CONTRACT = OutputContract(
    name="mitre",
    schema=MITRE_SCHEMA,
    required_fields=("technique_id", "tactic", "detection", "mitigation"),
)

ACTOR_CONTRACT = OutputContract(
    name="actor",
    schema=ACTOR_SCHEMA,
    required_fields=("actor_name", "ttps", "attribution_confidence"),
)

IOC_CONTRACT = OutputContract(
    name="ioc",
    schema=IOC_SCHEMA,
    required_fields=("ioc_type", "value", "confidence", "source_refs"),
)

__all__ = [
    "CVE_SCHEMA",
    "MITRE_SCHEMA",
    "ACTOR_SCHEMA",
    "IOC_SCHEMA",
    "THREAT_INTEL_GUARDRAILS",
    "CVE_CONTRACT",
    "MITRE_CONTRACT",
    "ACTOR_CONTRACT",
    "IOC_CONTRACT",
]
