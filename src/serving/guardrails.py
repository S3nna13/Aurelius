"""Runtime safety guardrails for Aurelius."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GuardrailPolicy:
    max_length: int = 4096
    block_patterns: List[str] = field(default_factory=list)
    harm_threshold: float = 0.8
    allow_adult: bool = False


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    modified_input: Optional[str] = None
    harm_score: float = 0.0


_SUSPICIOUS_SUBSTRINGS = ["INJECT", "OVERRIDE", "SYSTEM_BYPASS"]


class ContentGuardrails:
    def __init__(self, policy: GuardrailPolicy = None):
        self.policy = policy if policy is not None else GuardrailPolicy()

    def _pattern_check(self, text: str) -> Tuple[bool, str]:
        for pattern in self.policy.block_patterns:
            if pattern in text:
                return True, pattern
        return False, ""

    def _length_check(self, text: str) -> bool:
        return len(text) <= self.policy.max_length

    def _harm_score(self, text: str) -> float:
        if not _SUSPICIOUS_SUBSTRINGS:
            return 0.0
        matched = sum(1 for s in _SUSPICIOUS_SUBSTRINGS if s in text)
        return matched / len(_SUSPICIOUS_SUBSTRINGS)

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
