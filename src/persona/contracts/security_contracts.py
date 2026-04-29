from ..unified_persona import OutputContract

FINDING_CONTRACT = OutputContract(
    name="finding",
    schema={
        "type": "object",
        "required": ["id", "name", "severity", "affected_asset", "proof_of_concept_steps", "remediation"],
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "affected_asset": {"type": "string"},
            "proof_of_concept_steps": {"type": "array", "items": {"type": "string"}},
            "remediation": {"type": "string"},
        },
    },
    required_fields=("id", "name", "severity", "remediation"),
)

ALERT_CONTRACT = OutputContract(
    name="alert",
    schema={
        "type": "object",
        "required": ["id", "severity", "indicators", "mitre_mapping", "recommended_actions"],
        "properties": {
            "id": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "indicators": {"type": "array", "items": {"type": "object"}},
            "mitre_mapping": {"type": "array", "items": {"type": "string"}},
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("id", "severity", "indicators", "mitre_mapping"),
)

EMULATION_CONTRACT = OutputContract(
    name="emulation",
    schema={
        "type": "object",
        "required": ["ttp_id", "mitre_technique", "gap_identified", "priority"],
        "properties": {
            "ttp_id": {"type": "string"},
            "mitre_technique": {"type": "string"},
            "detection_rule_proposed": {"type": "string"},
            "gap_identified": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        },
    },
    required_fields=("ttp_id", "mitre_technique", "gap_identified", "priority"),
)
