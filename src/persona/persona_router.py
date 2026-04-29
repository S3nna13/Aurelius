from __future__ import annotations

import re
from typing import Any

from .unified_persona import PersonaDomain, UnifiedPersona

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

_SECURITY_KEYWORDS = frozenset([
    "vulnerability", "cve", "cwe", "exploit", "pentest",
    "red team", "blue team", "purple team",
    "soc", "malware", "ransomware", "phishing", "ioc", "ttp",
])

_PERSONALITY_KEYWORDS: dict[str, str] = {
    "aurelius-architect": "architect",
    "aurelius-coding": "detective",
    "aurelius-general": "guardian",
    "aurelius-analyst": "analyst",
    "aurelius-teacher": "teacher",
}

_PERSONALITY_WORDS: dict[str, list[str]] = {
    "architect": ["design", "build", "implement", "create", "develop", "refactor", "architecture", "structure", "pattern", "scaffold", "compose", "plan"],
    "detective": ["debug", "fix", "investigate", "trace", "diagnose", "bug", "error", "issue", "broken", "crash", "reproduce", "locate"],
    "guardian": ["security", "audit", "maintain", "vulnerability", "compliance", "harden", "monitor", "backup", "recovery", "upgrade", "dependency", "permissions"],
    "analyst": ["analyze", "evaluate", "measure", "benchmark", "profile", "compare", "metric", "statistics", "report", "assess", "quantify", "inspect"],
    "teacher": ["explain", "document", "guide", "teach", "tutorial", "describe", "walkthrough", "comment", "clarify", "onboard", "introduction", "example"],
}


class PersonaRouter:
    def __init__(self, registry: Any | None = None) -> None:
        self.registry = registry

    def route(self, user_input: str) -> UnifiedPersona:
        if self.registry is None:
            from .builtins import AURELIUS_GENERAL
            return AURELIUS_GENERAL

        if _CVE_RE.search(user_input) or _MITRE_RE.search(user_input):
            return self.registry.get("aurelius-threatintel")

        if self._security_keywords_present(user_input):
            return self._route_security(user_input)

        scores = self._score_all(user_input)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return self.registry.get(best)

        return self.registry.get("aurelius-general")

    def _security_keywords_present(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in _SECURITY_KEYWORDS)

    def _route_security(self, text: str) -> UnifiedPersona:
        text_lower = text.lower()
        if "purple team" in text_lower:
            return self.registry.get("aurelius-purpleteam")
        if "red team" in text_lower or "offensive" in text_lower or "exploit" in text_lower:
            return self.registry.get("aurelius-redteam")
        if "blue team" in text_lower or "soc" in text_lower or "incident response" in text_lower:
            return self.registry.get("aurelius-blueteam")
        return self.registry.get("aurelius-general")

    def _score_all(self, text: str) -> dict[str, int]:
        text_lower = text.lower()
        raw: dict[str, int] = {}
        for category, keywords in _PERSONALITY_WORDS.items():
            raw[category] = sum(1 for w in keywords if w in text_lower)
        scores: dict[str, int] = {}
        for persona_id, category in _PERSONALITY_KEYWORDS.items():
            scores[persona_id] = raw.get(category, 0)
        return scores
