"""Security persona output contracts — finding, alert, and emulation JSON schemas.

Extracted from src/chat/security_personas.py to serve the unified persona system.
"""

from __future__ import annotations

from typing import Any

from ..unified_persona import OutputContract

RED_TEAM_OUTPUT_CONTRACT: dict[str, Any] = {
    "finding": {
        "id": "string",
        "name": "string",
        "severity": "low|medium|high|critical",
        "affected_asset": "string",
        "proof_of_concept_steps": "list[string]",
        "remediation": "string",
    }
}

BLUE_TEAM_OUTPUT_CONTRACT: dict[str, Any] = {
    "alert": {
        "id": "string",
        "severity": "low|medium|high|critical",
        "indicators": "list[{type,value,evidence}]",
        "mitre_mapping": "list[string]",
        "recommended_actions": "list[string]",
        "runbook_ref": "string",
    }
}

PURPLE_TEAM_OUTPUT_CONTRACT: dict[str, Any] = {
    "emulation": {
        "ttp_id": "string",
        "mitre_technique": "string",
        "detection_rule_proposed": "string",
        "gap_identified": "string",
        "priority": "low|medium|high|critical",
    }
}

SECURITY_FINDING_CONTRACT = OutputContract(
    name="finding",
    schema=RED_TEAM_OUTPUT_CONTRACT,
    required_fields=("id", "name", "severity", "affected_asset", "proof_of_concept_steps", "remediation"),
)

SECURITY_ALERT_CONTRACT = OutputContract(
    name="alert",
    schema=BLUE_TEAM_OUTPUT_CONTRACT,
    required_fields=("id", "severity", "indicators", "mitre_mapping", "recommended_actions"),
)

SECURITY_EMULATION_CONTRACT = OutputContract(
    name="emulation",
    schema=PURPLE_TEAM_OUTPUT_CONTRACT,
    required_fields=("ttp_id", "mitre_technique", "detection_rule_proposed", "gap_identified", "priority"),
)

SECURITY_GUARDRAILS = {
    "red_team": (
        "Only operate on assets the user has identified as authorized / internal test targets.",
        "Refuse to operate on assets outside the declared scope.",
        "Never emit working exploit code aimed at third-party production systems.",
        "When asked about real-world targets, pivot to defensive posture.",
    ),
    "blue_team": (
        "Prioritize verified observables.",
        "Never fabricate IOCs.",
        "Cite evidence source (log line / packet / hash) for every claim.",
        "Escalate destructive actions to human approval.",
    ),
    "purple_team": (
        "Only operate on assets the user has identified as authorized / internal test targets.",
        "Refuse to operate on assets outside the declared scope.",
        "Never emit working exploit code aimed at third-party production systems.",
        "When asked about real-world targets, pivot to defensive posture.",
        "Prioritize verified observables.",
        "Never fabricate IOCs.",
        "Cite evidence source (log line / packet / hash) for every claim.",
        "Escalate destructive actions to human approval.",
    ),
}

__all__ = [
    "RED_TEAM_OUTPUT_CONTRACT",
    "BLUE_TEAM_OUTPUT_CONTRACT",
    "PURPLE_TEAM_OUTPUT_CONTRACT",
    "SECURITY_FINDING_CONTRACT",
    "SECURITY_ALERT_CONTRACT",
    "SECURITY_EMULATION_CONTRACT",
    "SECURITY_GUARDRAILS",
]