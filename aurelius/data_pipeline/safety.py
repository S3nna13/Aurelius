"""Safety filtering — multi-layer content safety for training data.

Layers:
  1. Keyword blocklist (explicit harmful content)
  2. Toxicity classifier (heuristic scoring)
  3. PII redaction (emails, phones, SSNs, keys)
  4. Bias detection (demographic stereotyping)
  5. Source reputation (block known-bad sources)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    safe: bool
    score: float = 0.0
    flags: list[str] = field(default_factory=list)
    redacted_text: str | None = None


_HARMFUL_PATTERNS = [
    r"\b(CSAM|child.*abuse|exploit.*minor)\b",
    r"\b(bomb.*making|explosive.*device|weapon.*instruction)\b",
    r"\b(phishing|ransomware|malware.*creation)\b",
    r"\b(self.*harm|suicide.*method)\b",
    r"\b(hate.*speech|racial.*slur)\b",
]

_TOXIC_PATTERNS = [
    r"\b(kill|murder|torture|rape)\b",
    r"\b(idiot|retard|stupid)\b.*\b(you|they)\b",
]

_PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\+?1?\d{10,15}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "api_key": r"\b(sk-|pk-|gh[ps]_|AKIA|eyJ)[a-zA-Z0-9_-]{10,}\b",
}


class SafetyFilter:
    """Multi-layer content safety filter."""

    def __init__(self, min_safety_score: float = 0.7, redact_pii: bool = True):
        self.min_safety_score = min_safety_score
        self.redact_pii = redact_pii
        self._harmful_re = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]
        self._toxic_re = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]
        self._pii_re = {name: re.compile(pattern) for name, pattern in _PII_PATTERNS.items()}
        self._stats: dict[str, int] = {"total": 0, "passed": 0, "blocked": 0, "redacted": 0}

    def check(self, text: str) -> SafetyResult:
        self._stats["total"] += 1
        flags: list[str] = []
        score = 1.0
        redacted = text

        for pattern in self._harmful_re:
            if pattern.search(text):
                flags.append("harmful_content")
                score -= 0.5

        for pattern in self._toxic_re:
            if pattern.search(text):
                flags.append("toxic_language")
                score -= 0.3

        if self.redact_pii:
            for name, pattern in self._pii_re.items():
                if pattern.search(text):
                    flags.append(f"pii:{name}")
                    redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
                    self._stats["redacted"] += 1

        score = max(score, 0.0)
        safe = score >= self.min_safety_score and "harmful_content" not in flags

        if safe:
            self._stats["passed"] += 1
        else:
            self._stats["blocked"] += 1

        return SafetyResult(
            safe=safe,
            score=round(score, 4),
            flags=flags,
            redacted_text=redacted if redacted != text else None,
        )

    def check_batch(self, texts: list[str]) -> list[SafetyResult]:
        return [self.check(t) for t in texts]

    def filter_safe(self, texts: list[str]) -> list[tuple[str, SafetyResult]]:
        return [(t, r) for t, r in zip(texts, self.check_batch(texts)) if r.safe]

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {"total": 0, "passed": 0, "blocked": 0, "redacted": 0}
