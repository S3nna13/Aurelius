"""Runtime safety guardrails for Aurelius."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GuardrailPolicy:
    max_length: int = 4096
    block_patterns: list[str] = field(default_factory=list)
    harm_threshold: float = 0.5
    allow_adult: bool = False


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    modified_input: str | None = None
    harm_score: float = 0.0


# Regex patterns — harder to bypass than plain substring matching
_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?i)\binject\b"),
    re.compile(r"(?i)\boverride\b"),
    re.compile(r"(?i)\bsystem.?bypass\b"),
    re.compile(r"(?i)\bigmore\s+(above|below|all|previous|following)\b"),
    re.compile(r"(?i)\bdisregard\b"),
    re.compile(r"(?i)\bprint.*flag\b"),
]


class ContentGuardrails:
    def __init__(self, policy: GuardrailPolicy | None = None):
        self.policy = policy if policy is not None else GuardrailPolicy()
        self._block_regexes = [re.compile(p, re.IGNORECASE) for p in self.policy.block_patterns]

    def _pattern_check(self, text: str) -> tuple[bool, str]:
        for regex in self._block_regexes + _SUSPICIOUS_PATTERNS:
            match = regex.search(text)
            if match:
                return True, match.re.pattern
        return False, ""

    def _length_check(self, text: str) -> bool:
        return len(text) <= self.policy.max_length

    def _harm_score(self, text: str) -> float:
        matches = sum(1 for p in _SUSPICIOUS_PATTERNS if p.search(text))
        return min(1.0, matches / len(_SUSPICIOUS_PATTERNS))

    def check_input(self, text: str) -> GuardrailResult:
        if not self._length_check(text):
            return GuardrailResult(
                allowed=False,
                reason="Input exceeds maximum length",
                harm_score=self._harm_score(text),
            )

        blocked, pattern = self._pattern_check(text)
        if blocked:
            return GuardrailResult(
                allowed=False,
                reason=f"Blocked pattern matched: {pattern}",
                harm_score=self._harm_score(text),
            )

        score = self._harm_score(text)
        if score >= self.policy.harm_threshold:
            return GuardrailResult(
                allowed=False,
                reason="Harm score exceeds threshold",
                harm_score=score,
            )

        return GuardrailResult(allowed=True, reason="OK", harm_score=score)

    def check_output(self, text: str) -> GuardrailResult:
        if not self._length_check(text):
            return GuardrailResult(
                allowed=False,
                reason="Output exceeds maximum length",
                harm_score=self._harm_score(text),
            )

        blocked, pattern = self._pattern_check(text)
        if blocked:
            return GuardrailResult(
                allowed=False,
                reason=f"Blocked pattern matched: {pattern}",
                harm_score=self._harm_score(text),
            )

        score = self._harm_score(text)
        if score >= self.policy.harm_threshold:
            return GuardrailResult(
                allowed=False,
                reason="Harm score exceeds threshold",
                harm_score=score,
            )

        return GuardrailResult(allowed=True, reason="OK", harm_score=score)

    def truncate_if_needed(self, text: str) -> str:
        if len(text) > self.policy.max_length:
            return text[: self.policy.max_length]
        return text
