"""Weighted-keyword severity triage for vulnerability-like text.

Stdlib-only, deterministic. Adapted from the bugbounty-agent reference
triage module, reduced to a pure heuristic scorer (no ML dependency) that
returns structured, frozen ``TriageResult`` records.

The scorer sums matched keyword weights, normalises by the maximum
possible weight sum (capped at 1.0), and maps the normalised score to a
``SeverityLevel`` via fixed thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SeverityLevel(Enum):
    CRITICAL = ("critical", 1.0)
    HIGH = ("high", 0.8)
    MEDIUM = ("medium", 0.5)
    LOW = ("low", 0.3)
    INFO = ("info", 0.1)

    def __init__(self, label: str, weight: float) -> None:
        self.label = label
        self.weight = weight


WEIGHTED_KEYWORDS: list[tuple[str, float]] = [
    ("rce", 1.0),
    ("remote code execution", 1.0),
    ("buffer overflow", 1.0),
    ("sql injection", 0.9),
    ("sqli", 0.9),
    ("command injection", 0.9),
    ("deserialization", 0.9),
    ("xxe", 0.9),
    ("xml external entity", 0.9),
    ("ssrf", 0.8),
    ("auth bypass", 0.8),
    ("authentication bypass", 0.8),
    ("privilege escalation", 0.8),
    ("lfi", 0.8),
    ("rfi", 0.8),
    ("local file inclusion", 0.8),
    ("remote file inclusion", 0.8),
    ("xss", 0.7),
    ("cross-site scripting", 0.7),
    ("idor", 0.7),
    ("path traversal", 0.7),
    ("directory traversal", 0.7),
    ("csrf", 0.5),
    ("information disclosure", 0.4),
    ("info disclosure", 0.4),
    ("dos", 0.4),
    ("misconfiguration", 0.4),
    ("weak crypto", 0.4),
    ("weak cryptography", 0.4),
    ("hardcoded", 0.4),
    ("open redirect", 0.3),
]


_MAX_WEIGHT = sum(w for _, w in WEIGHTED_KEYWORDS)


@dataclass(frozen=True)
class TriageResult:
    text: str
    score: float
    severity: SeverityLevel
    matched_keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0


def _severity_for_score(score: float) -> SeverityLevel:
    if score >= 0.7:
        return SeverityLevel.CRITICAL
    if score >= 0.5:
        return SeverityLevel.HIGH
    if score >= 0.3:
        return SeverityLevel.MEDIUM
    if score >= 0.1:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


class SeverityTriage:
    """Weighted-keyword severity scorer."""

    def __init__(self, keywords: list[tuple[str, float]] | None = None) -> None:
        self._keywords = list(keywords) if keywords is not None else list(WEIGHTED_KEYWORDS)
        self._max_weight = sum(w for _, w in self._keywords) or 1.0

    def score(self, text: str) -> TriageResult:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        lower = text.lower()
        matched: list[str] = []
        hit_weight = 0.0
        for kw, w in self._keywords:
            if kw in lower:
                matched.append(kw)
                hit_weight += w
        normalized = min(1.0, hit_weight / self._max_weight) if self._max_weight else 0.0
        severity = _severity_for_score(normalized)
        # Confidence rises with number of distinct matches, saturating at 5 hits.
        confidence = min(1.0, len(matched) / 5.0) if matched else 0.0
        return TriageResult(
            text=text,
            score=normalized,
            severity=severity,
            matched_keywords=matched,
            confidence=confidence,
        )

    def score_batch(self, texts: list[str]) -> list[TriageResult]:
        return [self.score(t) for t in texts]

    def top_k(self, texts: list[str], k: int) -> list[TriageResult]:
        if k <= 0:
            return []
        results = self.score_batch(texts)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


SEVERITY_TRIAGE_REGISTRY = {"default": SeverityTriage}
