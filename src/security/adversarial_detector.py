"""Multi-pattern adversarial input detector (defense layer).

Covers jailbreak attempts, prompt leakage, indirect injection,
encoding obfuscation, repetition flooding, and GCG-style gradient suffixes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class AdversarialPattern(StrEnum):
    JAILBREAK = "jailbreak"
    PROMPT_LEAK = "prompt_leak"
    INDIRECT_INJECTION = "indirect"
    ENCODING_ATTACK = "encoding"
    REPETITION_ATTACK = "repetition"
    GRADIENT_ATTACK = "gradient"


@dataclass
class AdversarialResult:
    text: str
    patterns_detected: list[AdversarialPattern]
    risk_score: float
    flagged: bool
    details: dict[str, str]


_JAILBREAK_RE = re.compile(
    r"\bDAN\b"
    r"|ignore\s+previous"
    r"|you\s+are\s+now\b"
    r"|pretend\s+you\s+are\b"
    r"|roleplay\s+as\b",
    re.IGNORECASE,
)

_PROMPT_LEAK_RE = re.compile(
    r"repeat\s+your\s+system"
    r"|show\s+me\s+your\s+prompt"
    r"|what\s+are\s+your\s+instructions",
    re.IGNORECASE,
)

_INDIRECT_RE = re.compile(
    r"https?://\S+\s+(follow|execute|run)\b"
    r"|\[\[inject",
    re.IGNORECASE,
)

_B64_CHUNK_RE = re.compile(r"[A-Za-z0-9+/=]{40,}")

_GCG_CHUNK_RE = re.compile(r"\S{30,}")


def _has_unicode_controls(text: str) -> bool:
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cc") and ch not in ("\n", "\r", "\t"):
            return True
    return False


def _repetition_score(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if len(sentences) < 5:
        return 0.0
    counts: dict[str, int] = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    max_repeat = max(counts.values())
    return 1.0 if max_repeat >= 5 else 0.0


class AdversarialDetector:
    """Multi-pattern adversarial input detector (defense layer)."""

    _WEIGHTS: dict[AdversarialPattern, float] = {
        AdversarialPattern.JAILBREAK: 0.35,
        AdversarialPattern.PROMPT_LEAK: 0.25,
        AdversarialPattern.INDIRECT_INJECTION: 0.20,
        AdversarialPattern.ENCODING_ATTACK: 0.15,
        AdversarialPattern.REPETITION_ATTACK: 0.20,
        AdversarialPattern.GRADIENT_ATTACK: 0.15,
    }

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def detect(self, text: str) -> AdversarialResult:
        detected: list[AdversarialPattern] = []
        details: dict[str, str] = {}
        raw_score = 0.0

        if _JAILBREAK_RE.search(text):
            p = AdversarialPattern.JAILBREAK
            detected.append(p)
            details[p.value] = "matched jailbreak pattern"
            raw_score += self._WEIGHTS[p]

        if _PROMPT_LEAK_RE.search(text):
            p = AdversarialPattern.PROMPT_LEAK
            detected.append(p)
            details[p.value] = "matched prompt-leak pattern"
            raw_score += self._WEIGHTS[p]

        if _INDIRECT_RE.search(text):
            p = AdversarialPattern.INDIRECT_INJECTION
            detected.append(p)
            details[p.value] = "matched indirect injection pattern"
            raw_score += self._WEIGHTS[p]

        encoding_reasons: list[str] = []
        if _B64_CHUNK_RE.search(text):
            encoding_reasons.append("base64-like chunk")
        if _has_unicode_controls(text):
            encoding_reasons.append("unicode control chars")
        if encoding_reasons:
            p = AdversarialPattern.ENCODING_ATTACK
            detected.append(p)
            details[p.value] = "; ".join(encoding_reasons)
            raw_score += self._WEIGHTS[p]

        rep = _repetition_score(text)
        if rep > 0:
            p = AdversarialPattern.REPETITION_ATTACK
            detected.append(p)
            details[p.value] = "sentence repeated 5+ times"
            raw_score += self._WEIGHTS[p]

        gcg_hits = [
            m.group()
            for m in _GCG_CHUNK_RE.finditer(text)
            if not re.match(r"https?://", m.group()) and not _B64_CHUNK_RE.fullmatch(m.group())
        ]
        if gcg_hits:
            p = AdversarialPattern.GRADIENT_ATTACK
            detected.append(p)
            details[p.value] = f"GCG-style token(s): {gcg_hits[0][:40]!r}"
            raw_score += self._WEIGHTS[p]

        risk_score = min(raw_score, 1.0)
        return AdversarialResult(
            text=text,
            patterns_detected=detected,
            risk_score=risk_score,
            flagged=risk_score > self._threshold,
            details=details,
        )

    def batch_detect(self, texts: list[str]) -> list[AdversarialResult]:
        return [self.detect(t) for t in texts]

    def update_threshold(self, threshold: float) -> None:
        self._threshold = threshold
