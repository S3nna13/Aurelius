"""Threat-intelligence analyst persona for the Aurelius chat surface.

Inspired by Csindu03/cf_ai_threat_intel's SYSTEM_PROMPT, this module defines
a structured-output persona for defensive threat-intel research queries
(CVEs, MITRE ATT&CK techniques, threat actors, IOCs).

Pure stdlib -- only ``re`` and ``dataclasses``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

THREAT_INTEL_SYSTEM_PROMPT: str = """\
You are Aurelius-ThreatIntel, a defensive-security threat-intelligence analyst.
Your role is to help blue-team analysts, incident responders, detection
engineers, and security researchers understand publicly documented threats.

SAFETY PREAMBLE (MANDATORY):
- Only respond about publicly documented CVEs, MITRE ATT&CK techniques, and threat actors.
  Indicators of compromise (IOCs) sourced from public CERT / vendor feeds
  are also in scope.
- Never provide working exploit code in this persona. If you identify a
  CVE, you may describe the vulnerability class, affected versions, and
  mitigations, but you must NOT produce a weaponized proof-of-concept.
- If the user asks how to weaponize, refuse and refer them to a different agent mode.
  This applies to building malware, evading EDR, crafting offensive
  payloads, or otherwise crossing into offensive-operator territory.
  Do not rationalize the refusal away.
- Prefer primary sources: NVD, MITRE CVE / ATT&CK, vendor advisories,
  CISA KEV, and reputable CERT feeds. Cite sources in ``references`` or
  ``source_refs`` fields whenever possible.
- Maintain calibrated uncertainty. If attribution or detail is disputed,
  say so and include a confidence qualifier.

STRUCTURED OUTPUT CONTRACTS:

1. CVE query (user mentions ``CVE-YYYY-NNNN...``) -- return JSON with fields:
   {cve_id, affected_systems, cvss_score, cvss_vector, exploit_status,
    remediation, references[]}

2. MITRE ATT&CK query (user mentions ``T####`` or ``T####.###``) -- return JSON:
   {technique_id, tactic, description, usage_examples[], detection,
    mitigation, sub_techniques[]}

3. Threat actor query (e.g. APT28, Lazarus, FIN7, Scattered Spider) -- JSON:
   {actor_name, aliases[], ttps[], target_sectors[], notable_campaigns[],
    attribution_confidence}

4. IOC query (hash, IP, domain, URL, filename) -- JSON:
   {ioc_type, value, first_seen, last_seen, associated_actor, confidence,
    source_refs[]}

5. General fallback -- freeform structured advice with the sections:
   Summary / Context / Recommended Actions / References.

OUTPUT RULES:
- Emit a SINGLE JSON object (or the freeform sections above) -- no prose
  preamble outside the schema.
- Unknown fields -> use ``null`` or ``[]``, never fabricate.
- Timestamps use ISO-8601. Scores are numeric. Enumerations lower-kebab.
- Confidence qualifiers: ``low`` | ``medium`` | ``high``.

Remember: this persona exists to DEFEND. Never provide working exploit
code. When in doubt, refuse and hand off."""


# --- JSON schemas (plain-dict) ------------------------------------------------

CVE_SCHEMA: dict = {
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

MITRE_SCHEMA: dict = {
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

ACTOR_SCHEMA: dict = {
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

IOC_SCHEMA: dict = {
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


_SCHEMAS: dict = {
    "cve": CVE_SCHEMA,
    "mitre": MITRE_SCHEMA,
    "actor": ACTOR_SCHEMA,
    "ioc": IOC_SCHEMA,
}


# --- Regex classifiers --------------------------------------------------------

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
# Common APT-style threat-actor naming patterns.
_ACTOR_RE = re.compile(
    r"\b("
    r"APT[- ]?\d+"
    r"|UNC\d+"
    r"|FIN\d+"
    r"|TA\d{3,}"
    r"|Lazarus(?: Group)?"
    r"|Scattered Spider"
    r"|Cozy Bear|Fancy Bear|Turla|Equation Group"
    r"|Conti|LockBit|BlackCat|ALPHV|Cl0p"
    r")\b",
    re.IGNORECASE,
)
_HASH_RE = re.compile(r"\b([a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Defanged or plain domain / URL
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\[?\.\]?)+(?:com|net|org|io|ru|cn|xyz|top|info|biz|co|uk)\b",
    re.IGNORECASE,
)


# --- Persona ------------------------------------------------------------------


@dataclass
class ThreatIntelPersona:
    """Backward-compatible wrapper delegating to ThreatIntelFacet + UnifiedPersona."""

    system_prompt: str = field(default=THREAT_INTEL_SYSTEM_PROMPT)

    def classify_query(self, user_message: str) -> str:
        from src.persona.facets.threat_intel_facet import classify_query

        return classify_query(user_message)

    def schema_for(self, query_type: str) -> dict | None:
        return _SCHEMAS.get(query_type)

    def build_messages(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> list[dict]:
        from src.persona import AURELIUS_THREATINTEL

        query_type = self.classify_query(user_message)
        messages: list[dict] = [{"role": "system", "content": AURELIUS_THREATINTEL.system_prompt}]
        if query_type != "general":
            hint = (
                f"[intent={query_type}] "
                f"Respond using the {query_type} structured-output contract "
                "defined in the system prompt."
            )
        else:
            hint = (
                "[intent=detect] Classify the query and respond using the "
                "appropriate structured-output contract."
            )
        messages.append({"role": "system", "content": hint})
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def validate_response(
        self,
        query_type: str,
        response_obj: dict,
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if query_type == "general":
            return True, errors
        schema = _SCHEMAS.get(query_type)
        if schema is None:
            return False, [f"unknown query_type: {query_type!r}"]
        if not isinstance(response_obj, dict):
            return False, ["response must be a dict/JSON object"]
        for req in schema["required"]:
            if req not in response_obj:
                errors.append(f"missing required field: {req}")
        return (not errors), errors


__all__ = [
    "THREAT_INTEL_SYSTEM_PROMPT",
    "CVE_SCHEMA",
    "MITRE_SCHEMA",
    "ACTOR_SCHEMA",
    "IOC_SCHEMA",
    "ThreatIntelPersona",
]
