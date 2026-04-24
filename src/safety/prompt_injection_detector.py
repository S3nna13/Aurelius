"""Prompt Injection Detector — detects adversarial injection attempts.

Uses regex-based signal detection plus a base64 payload heuristic.
No external ML frameworks required.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class InjectionSignal(str, Enum):
    ROLE_OVERRIDE = "role_override"
    INSTRUCTION_IGNORE = "instruction_ignore"
    SYSTEM_LEAK = "system_leak"
    DELIM_INJECT = "delim_inject"
    BASE64_PAYLOAD = "base64_payload"


@dataclass
class InjectionResult:
    text: str
    signals: List[InjectionSignal]
    score: float
    blocked: bool


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

_SIGNAL_PATTERNS: dict = {
    InjectionSignal.ROLE_OVERRIDE: re.compile(
        r"(?i)(you are now|act as|pretend you are|ignore previous|new persona)"
    ),
    InjectionSignal.INSTRUCTION_IGNORE: re.compile(
        r"(?i)(ignore (all )?previous|disregard|forget (your )?instructions)"
    ),
    InjectionSignal.SYSTEM_LEAK: re.compile(
        r"(?i)(print (your )?system prompt|reveal (your )?instructions"
        r"|what (are|were) your instructions)"
    ),
    InjectionSignal.DELIM_INJECT: re.compile(
        r"(\[INST\]|<\|system\|>|<\|user\|>|<<SYS>>|\[/INST\])"
    ),
}

# Matches candidate base64 tokens (word chars + /+=, length > 50)
_B64_CANDIDATE = re.compile(r"[A-Za-z0-9+/=]{51,}")


def _is_ascii_instruction(decoded: str) -> bool:
    """Return True if decoded string looks like an ASCII instruction."""
    printable_ratio = sum(32 <= ord(c) < 127 for c in decoded) / max(len(decoded), 1)
    return printable_ratio > 0.85 and len(decoded) > 10


class PromptInjectionDetector:
    """Detects prompt injection signals in user-supplied text."""

    def detect(self, text: str) -> InjectionResult:
        """Analyse text for injection signals.

        Args:
            text: Input text to inspect.

        Returns:
            InjectionResult with detected signals, composite score, and
            blocked flag (score >= 0.3).
        """
        signals: List[InjectionSignal] = []

        # Regex-based signals
        for signal, pattern in _SIGNAL_PATTERNS.items():
            if pattern.search(text):
                signals.append(signal)

        # Base64 payload detection
        for candidate in _B64_CANDIDATE.findall(text):
            # Pad to multiple of 4
            padded = candidate + "=" * (-len(candidate) % 4)
            try:
                decoded_bytes = base64.b64decode(padded)
                decoded_str = decoded_bytes.decode("ascii", errors="replace")
                if _is_ascii_instruction(decoded_str):
                    if InjectionSignal.BASE64_PAYLOAD not in signals:
                        signals.append(InjectionSignal.BASE64_PAYLOAD)
            except (binascii.Error, ValueError):
                pass

        # Score: fraction of possible signals triggered
        score = len(signals) / len(InjectionSignal)
        blocked = score >= 0.3

        return InjectionResult(
            text=text,
            signals=signals,
            score=score,
            blocked=blocked,
        )

    def is_safe(self, text: str, threshold: float = 0.3) -> bool:
        """Return True if the text is considered safe (no injection detected).

        Args:
            text: Input text.
            threshold: Score threshold above which the text is considered unsafe.

        Returns:
            True if safe (score < threshold).
        """
        result = self.detect(text)
        return result.score < threshold


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAFETY_REGISTRY: dict = {}
SAFETY_REGISTRY["prompt_injection_detector"] = PromptInjectionDetector()
