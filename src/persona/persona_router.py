"""Unified persona router — replaces PersonalityRouter + IntentClassifier + ThreatIntelPersona.classify_query.

Routes user input to the best UnifiedPersona using:
1. Domain-specific classifiers (CVE patterns, MITRE IDs, security keywords)
2. Keyword scoring across all personas
3. Intent classification fallback
4. Default persona
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .persona_registry import UnifiedPersonaRegistry

from .unified_persona import PersonaDomain, UnifiedPersona


@dataclass(frozen=True)
class RoutingResult:
    persona_id: str
    domain: PersonaDomain
    confidence: float
    matched_keywords: list[str]
    matched_patterns: list[str]
    fallback_used: bool


_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
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
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\[?\.\]?)+(?:com|net|org|io|ru|cn|xyz|top|info|biz|co|uk)\b",
    re.IGNORECASE,
)

_SECURITY_KEYWORDS: list[str] = [
    "vulnerability", "exploit", "CVE", "pentest", "penetration test",
    "incident", "SOC", "SIEM", "firewall", "IDS", "IPS", "malware",
    "phishing", "ransomware", "breach", "threat actor", "IOC",
    "forensics", "hardening", "privilege escalation", "lateral movement",
    "exfiltration", "command and control", "C2", "zero day",
    "red team", "blue team", "purple team", "MITRE", "ATT&CK",
]

_PERSONA_KEYWORDS: dict[str, list[str]] = {
    "aurelius-coding": [
        "code", "implement", "function", "class", "method", "debug",
        "refactor", "compile", "runtime", "stack trace", "syntax",
        "variable", "module", "package", "import", "API",
    ],
    "aurelius-architect": [
        "design", "architecture", "system", "migrate", "trade-off",
        "plan", "structure", "pattern", "scale", "deploy",
    ],
    "aurelius-debug": [
        "debug", "trace", "error", "crash", "bug", "issue",
        "reproduce", "log", "stack", "fault", "diagnose",
    ],
    "aurelius-analyst": [
        "analyze", "evaluate", "measure", "benchmark", "profile",
        "compare", "metric", "statistics", "report", "assess",
    ],
    "aurelius-teacher": [
        "explain", "document", "guide", "teach", "tutorial",
        "walkthrough", "clarify", "onboard", "introduction",
    ],
}

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "question": ["what", "who", "where", "when", "why", "how", "which", "?"],
    "request": ["please", "can you", "could you", "i want", "i need", "give me"],
    "complaint": ["not working", "broken", "wrong", "issue", "problem", "error"],
}


class PersonaRouter:
    """Unified router that replaces PersonalityRouter + IntentClassifier.

    Routes user input to the best UnifiedPersona using domain-specific
    classifiers, keyword scoring, and intent fallback.
    """

    def __init__(self, registry: UnifiedPersonaRegistry) -> None:
        self.registry = registry
        self._keyword_weights: dict[str, dict[str, float]] = {}
        for pid, words in _PERSONA_KEYWORDS.items():
            self._keyword_weights[pid] = {w: 1.0 for w in words}

    def route(self, user_input: str) -> UnifiedPersona:
        result = self.analyze(user_input)
        return self.registry.get(result.persona_id)

    def analyze(self, user_input: str) -> RoutingResult:
        text = user_input.lower()

        if _CVE_RE.search(text) or _MITRE_RE.search(text):
            return RoutingResult(
                persona_id="aurelius-threatintel",
                domain=PersonaDomain.THREAT_INTEL,
                confidence=0.95,
                matched_keywords=[],
                matched_patterns=_extract_patterns(text),
                fallback_used=False,
            )

        if _ACTOR_RE.search(text):
            return RoutingResult(
                persona_id="aurelius-threatintel",
                domain=PersonaDomain.THREAT_INTEL,
                confidence=0.85,
                matched_keywords=[],
                matched_patterns=_extract_patterns(text),
                fallback_used=False,
            )

        if _HASH_RE.search(text) or _IP_RE.search(text) or _DOMAIN_RE.search(text):
            return RoutingResult(
                persona_id="aurelius-threatintel",
                domain=PersonaDomain.THREAT_INTEL,
                confidence=0.7,
                matched_keywords=[],
                matched_patterns=_extract_patterns(text),
                fallback_used=False,
            )

        security_hits = [w for w in _SECURITY_KEYWORDS if w.lower() in text]
        if len(security_hits) >= 2:
            if any(w in text for w in ["red team", "pentest", "penetration test", "exploit"]):
                return RoutingResult(
                    persona_id="aurelius-redteam",
                    domain=PersonaDomain.SECURITY,
                    confidence=0.8,
                    matched_keywords=security_hits,
                    matched_patterns=[],
                    fallback_used=False,
                )
            if any(w in text for w in ["defend", "SOC", "incident", "detection", "monitor"]):
                return RoutingResult(
                    persona_id="aurelius-blueteam",
                    domain=PersonaDomain.SECURITY,
                    confidence=0.8,
                    matched_keywords=security_hits,
                    matched_patterns=[],
                    fallback_used=False,
                )
            return RoutingResult(
                persona_id="aurelius-purpleteam",
                domain=PersonaDomain.SECURITY,
                confidence=0.6,
                matched_keywords=security_hits,
                matched_patterns=[],
                fallback_used=False,
            )

        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for pid, weights in self._keyword_weights.items():
            if pid not in self.registry:
                continue
            hits = [w for w in weights if w in text]
            if hits:
                scores[pid] = len(hits) / len(weights)
                matched[pid] = hits

        if scores:
            best_id = max(scores, key=scores.get)  # type: ignore[arg-type]
            persona = self.registry.get(best_id)
            return RoutingResult(
                persona_id=best_id,
                domain=persona.domain,
                confidence=scores[best_id],
                matched_keywords=matched.get(best_id, []),
                matched_patterns=[],
                fallback_used=False,
            )

        return RoutingResult(
            persona_id="aurelius-general",
            domain=PersonaDomain.GENERAL,
            confidence=0.0,
            matched_keywords=[],
            matched_patterns=[],
            fallback_used=True,
        )


def _extract_patterns(text: str) -> list[str]:
    patterns: list[str] = []
    for match in _CVE_RE.finditer(text):
        patterns.append(f"cve:{match.group()}")
    for match in _MITRE_RE.finditer(text):
        patterns.append(f"mitre:{match.group()}")
    for match in _ACTOR_RE.finditer(text):
        patterns.append(f"actor:{match.group()}")
    return patterns


__all__ = ["PersonaRouter", "RoutingResult"]