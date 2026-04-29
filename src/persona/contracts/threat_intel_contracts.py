from ..unified_persona import OutputContract

CVE_CONTRACT = OutputContract(
    name="cve",
    schema={
        "type": "object",
        "required": ["cve_id", "affected_systems", "cvss_score", "remediation"],
        "properties": {
            "cve_id": {"type": "string"},
            "affected_systems": {"type": "array", "items": {"type": "string"}},
            "cvss_score": {"type": "number"},
            "cvss_vector": {"type": "string"},
            "exploit_status": {"type": "string"},
            "remediation": {"type": "string"},
            "references": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("cve_id", "affected_systems", "cvss_score", "remediation"),
)

MITRE_CONTRACT = OutputContract(
    name="mitre",
    schema={
        "type": "object",
        "required": ["technique_id", "tactic", "description", "detection", "mitigation"],
        "properties": {
            "technique_id": {"type": "string"},
            "tactic": {"type": "string"},
            "description": {"type": "string"},
            "usage_examples": {"type": "array", "items": {"type": "string"}},
            "detection": {"type": "string"},
            "mitigation": {"type": "string"},
            "sub_techniques": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("technique_id", "tactic", "description", "detection", "mitigation"),
)

ACTOR_CONTRACT = OutputContract(
    name="actor",
    schema={
        "type": "object",
        "required": ["actor_name", "ttps", "attribution_confidence"],
        "properties": {
            "actor_name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "ttps": {"type": "array", "items": {"type": "string"}},
            "target_sectors": {"type": "array", "items": {"type": "string"}},
            "notable_campaigns": {"type": "array", "items": {"type": "string"}},
            "attribution_confidence": {"type": "string"},
        },
    },
    required_fields=("actor_name", "ttps", "attribution_confidence"),
)

IOC_CONTRACT = OutputContract(
    name="ioc",
    schema={
        "type": "object",
        "required": ["ioc_type", "value", "confidence", "source_refs"],
        "properties": {
            "ioc_type": {"type": "string"},
            "value": {"type": "string"},
            "first_seen": {"type": "string"},
            "last_seen": {"type": "string"},
            "associated_actor": {"type": "string"},
            "confidence": {"type": "string"},
            "source_refs": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("ioc_type", "value", "confidence", "source_refs"),
)
