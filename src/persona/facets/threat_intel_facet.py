"""Threat intelligence facet — CVE/MITRE/actor/IOC classification and response validation."""

from __future__ import annotations

import re
from typing import Any

from ..unified_persona import PersonaFacet

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_ACTOR_RE = re.compile(
    r"\b(APT[- ]?\d+|UNC\d+|FIN\d+|TA\d{3,}"
    r"|Lazarus(?: Group)?|Scattered Spider"
    r"|Cozy Bear|Fancy Bear|Turla|Equation Group"
    r"|Conti|LockBit|BlackCat|ALPHV|Cl0p)\b",
    re.IGNORECASE,
)
_HASH_RE = re.compile(r"\b([a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\[?\.\]?)+(?:com|net|org|io|ru|cn|xyz|top|info|biz|co|uk)\b",
    re.IGNORECASE,
)


def create_threat_intel_facet(auto_classify: bool = True) -> PersonaFacet:
    return PersonaFacet(
        facet_type="threat_intel",
        config={
            "auto_classify": auto_classify,
            "query_classifiers": ["cve", "mitre", "actor", "ioc"],
        },
    )


def classify_query(user_message: str) -> str:
    if not isinstance(user_message, str) or not user_message:
        return "general"
    if _CVE_RE.search(user_message):
        return "cve"
    if _MITRE_RE.search(user_message):
        return "mitre"
    if _ACTOR_RE.search(user_message):
        return "actor"
    if (
        _HASH_RE.search(user_message)
        or _IP_RE.search(user_message)
        or _DOMAIN_RE.search(user_message)
    ):
        return "ioc"
    return "general"


def validate_threat_intel_response(query_type: str, response_obj: Any) -> tuple[bool, list[str]]:
    from ..contracts.threat_intel_contracts import (
        ACTOR_SCHEMA,
        CVE_SCHEMA,
        IOC_SCHEMA,
        MITRE_SCHEMA,
    )

    schemas = {"cve": CVE_SCHEMA, "mitre": MITRE_SCHEMA, "actor": ACTOR_SCHEMA, "ioc": IOC_SCHEMA}

    if query_type == "general":
        return True, []

    schema = schemas.get(query_type)
    if schema is None:
        return False, [f"unknown query_type: {query_type!r}"]

    if not isinstance(response_obj, dict):
        return False, ["response must be a dict/JSON object"]

    errors: list[str] = []
    for field in schema["required"]:
        if field not in response_obj:
            errors.append(f"missing required field: {field}")
    return (not errors), errors
