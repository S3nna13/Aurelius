"""Jailbreak attempt detector — Aurelius safety module.

Detects jailbreak attempts in prompts via regex-based category signals and
a weighted risk score. Stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JailbreakCategory(Enum):
    ROLE_PLAY = "role_play"
    HYPOTHETICAL = "hypothetical"
    INJECTION = "injection"
    OBFUSCATION = "obfuscation"
    AUTHORITY_CLAIM = "authority_claim"
    DIRECT_REQUEST = "direct_request"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JailbreakSignal:
    """A single matched jailbreak signal."""
    category: JailbreakCategory
    pattern: str
    weight: float
    matched_text: str


# ---------------------------------------------------------------------------
# Default weights per category
# ---------------------------------------------------------------------------

_CATEGORY_WEIGHTS: dict[JailbreakCategory, float] = {
    JailbreakCategory.ROLE_PLAY: 2.0,
    JailbreakCategory.HYPOTHETICAL: 1.5,
    JailbreakCategory.INJECTION: 4.0,
    JailbreakCategory.OBFUSCATION: 3.0,
    JailbreakCategory.AUTHORITY_CLAIM: 3.5,
    JailbreakCategory.DIRECT_REQUEST: 2.5,
}

# Raw pattern strings (used for signal.pattern field)
_PATTERN_STRINGS: dict[JailbreakCategory, str] = {
    JailbreakCategory.ROLE_PLAY: (
        r"(pretend|act as|you are now|imagine you|roleplay|play the role)"
    ),
    JailbreakCategory.HYPOTHETICAL: (
        r"(hypothetically|in a world where|what if|suppose that|let's say)"
    ),
    JailbreakCategory.INJECTION: (
        r"(ignore (previous|all|prior)|forget your|new instruction|override|system prompt)"
    ),
    JailbreakCategory.OBFUSCATION: (
        r"(base64|rot13|encoded|decode this|translate.*to.*english)"
    ),
    JailbreakCategory.AUTHORITY_CLAIM: (
        r"(i am (a|an|the) (developer|admin|researcher|openai|anthropic)|authorized|permission granted)"
    ),
    JailbreakCategory.DIRECT_REQUEST: (
        r"(tell me how to|give me instructions|provide (a )?(step|guide)|explain how to make)"
    ),
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class JailbreakDetector:
    """Regex-based jailbreak attempt detector."""

    def __init__(self) -> None:
        # Compile each category pattern once, case-insensitive
        self._compiled: dict[JailbreakCategory, re.Pattern] = {
            category: re.compile(pattern, re.IGNORECASE)
            for category, pattern in _PATTERN_STRINGS.items()
        }

    def detect(self, prompt: str) -> List[JailbreakSignal]:
        """Return all matched jailbreak signals for the given prompt.

        Each pattern category can match at most once per prompt.
        """
        signals: List[JailbreakSignal] = []
        for category, rx in self._compiled.items():
            m = rx.search(prompt)
            if m is not None:
                signals.append(
                    JailbreakSignal(
                        category=category,
                        pattern=_PATTERN_STRINGS[category],
                        weight=_CATEGORY_WEIGHTS[category],
                        matched_text=m.group(0),
                    )
                )
        return signals

    def risk_score(self, signals: List[JailbreakSignal]) -> float:
        """Sum signal weights, clamped to [0.0, 10.0]."""
        total = sum(s.weight for s in signals)
        if total < 0.0:
            return 0.0
        if total > 10.0:
            return 10.0
        return total

    def is_jailbreak(self, prompt: str, threshold: float = 4.0) -> bool:
        """Return True if the prompt's risk score meets or exceeds threshold."""
        signals = self.detect(prompt)
        return self.risk_score(signals) >= threshold


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

JAILBREAK_DETECTOR_REGISTRY: dict[str, type] = {
    "default": JailbreakDetector,
}
