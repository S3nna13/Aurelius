from __future__ import annotations

import re
from typing import Any

from ..unified_persona import PersonaFacet

THREAT_INTEL_FACET = PersonaFacet(
    facet_type="threat_intel",
    config={"query_classifiers": ["cve", "mitre", "actor", "ioc"]},
)

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_ACTOR_RE = re.compile(r"\b(APT[- ]?\d+|UNC\d+|FIN\d+|TA\d{3,}|Lazarus(?: Group)?|Scattered Spider|Cozy Bear|Fancy Bear|Turla|Equation Group|Conti|LockBit|BlackCat|ALPHV|Cl0p)\b", re.IGNORECASE)
_HASH_RE = re.compile(r"\b([a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\[?\.\]?)+(?:com|net|org|io|ru|cn|xyz|top|info|biz|co|uk)\b", re.IGNORECASE)


def classify_query(user_message: str) -> str:
    if not isinstance(user_message, str) or not user_message:
        return "general"
    if _CVE_RE.search(user_message):
        return "cve"
    if _MITRE_RE.search(user_message):
        return "mitre"
    if _ACTOR_RE.search(user_message):
        return "actor"
    if _HASH_RE.search(user_message) or _IP_RE.search(user_message) or _DOMAIN_RE.search(user_message):
        return "ioc"
    return "general"
