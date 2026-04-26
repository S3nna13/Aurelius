"""Quantitative security posture scorer for agent outputs.

Evaluates text against STRIDE categories and common CWE patterns using
stdlib-only regex heuristics.  Produces a numeric score and per-category
breakdown so callers can gate, log, or escalate based on measurable risk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class PostureScore:
    overall: float  # 0.0 – 100.0, higher is safer
    spoofing: float
    tampering: float
    repudiation: float
    information_disclosure: float
    denial_of_service: float
    elevation_of_privilege: float
    findings: list[str]


class SecurityPostureScorer:
    """Score text for security posture using lightweight heuristics.

    This is *not* a replacement for deep static analysis; it is a fast,
    dependency-free pre-flight / post-flight gate suitable for agent
    outputs, user prompts, and tool results.
    """

    _PATTERNS: ClassVar[dict[str, list[str]]] = {
        "spoofing": [
            r"\b(phish|impersonat|spoof|fake\s+(login|cred|token|auth))",
            r"\bpretend\s+to\s+be\b",
            r"\bman.in.the.middle\b",
        ],
        "tampering": [
            r"\b(modif\w+|alter\w+|injection|sqli|xss|command\s*inject)",
            r"\bexec\s*\(|eval\s*\(|__import__\s*\(",
            r"\b(os\.|subprocess\.|sys\.)(call|run|popen|system)",
        ],
        "repudiation": [
            r"\b(delete\s+log|clear\s+history|hide\s+trail|no\s+audit)",
            r"\buntraceable\b",
        ],
        "information_disclosure": [
            r"\b(password|secret|key|token|api[_-]?key)\s*=\s*['\"]\S+",
            r"\b(leak|expose|dump\s+db|credit.?card|ssn)\b",
            r"\b(internal|private)\s+(ip|host|endpoint|url)",
        ],
        "denial_of_service": [
            r"\b(infinite\s+loop|while\s+True\s*:|fork\s*bomb|memory\s*leak)",
            r"\b(consume\s+all|exhaust|flood|spam|ddos)\b",
            r"\b(recursion\s+without\s+base\s*case|unbounded\s+growth)",
        ],
        "elevation_of_privilege": [
            r"\b(sudo|root|admin|privileged|setuid|chmod\s+777)\b",
            r"\b(escalat\w+|bypass\w+\s+(auth|check|restrict))\b",
            r"\b(run\s+as\s+(root|admin|system))\b",
        ],
    }

    def __init__(self, max_text_length: int = 16_384) -> None:
        self.max_text_length = max_text_length
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self._PATTERNS.items()
        }

    def score(self, text: str) -> PostureScore:
        """Return a :class:`PostureScore` for *text*.

        The overall score is the average of the six STRIDE category
        scores.  Each category starts at 100.0 and loses 25 points per
        pattern match (floor 0.0).
        """
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        findings: list[str] = []
        per_category: dict[str, float] = {}

        for cat, patterns in self._compiled.items():
            hits = 0
            for pat in patterns:
                if pat.search(text):
                    hits += 1
                    # Avoid adding duplicate findings for the same category
                    if hits == 1:
                        findings.append(f"{cat}: matched heuristic pattern")
            per_category[cat] = max(0.0, 100.0 - hits * 25.0)

        overall = sum(per_category.values()) / len(per_category)
        return PostureScore(
            overall=overall,
            spoofing=per_category["spoofing"],
            tampering=per_category["tampering"],
            repudiation=per_category["repudiation"],
            information_disclosure=per_category["information_disclosure"],
            denial_of_service=per_category["denial_of_service"],
            elevation_of_privilege=per_category["elevation_of_privilege"],
            findings=findings,
        )

    def gate(self, text: str, minimum_score: float = 70.0) -> tuple[bool, PostureScore]:
        """Return ``(passed, score)`` where *passed* is True when ``overall >= minimum_score``."""
        s = self.score(text)
        return s.overall >= minimum_score, s


#: Module-level registry.
POSTURE_SCORER_REGISTRY: dict[str, SecurityPostureScorer] = {
    "default": SecurityPostureScorer(),
}
