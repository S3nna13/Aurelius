"""Jailbreak classifier v2: multi-signal ensemble with confidence scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class JailbreakSignal(str, Enum):
    ROLE_PLAY = "role_play"
    IGNORE_INSTRUCTIONS = "ignore_instructions"
    HYPOTHETICAL = "hypothetical"
    PROMPT_INJECTION = "prompt_injection"
    TOKEN_SMUGGLING = "token_smuggling"


@dataclass
class JailbreakDetection:
    signals: list[JailbreakSignal]
    confidence: float
    is_jailbreak: bool
    explanation: str = ""


class JailbreakClassifierV2:
    SIGNAL_PATTERNS: ClassVar[dict[JailbreakSignal, list[str]]] = {
        JailbreakSignal.ROLE_PLAY: [
            "pretend you are",
            "act as",
            "you are now",
            "roleplay as",
            "imagine you are a",
        ],
        JailbreakSignal.IGNORE_INSTRUCTIONS: [
            "ignore previous",
            "disregard all",
            "forget your",
            "override your",
            "ignore all instructions",
        ],
        JailbreakSignal.HYPOTHETICAL: [
            "hypothetically",
            "in a fictional",
            "for a story",
            "if you were not",
            "theoretically speaking",
        ],
        JailbreakSignal.PROMPT_INJECTION: [
            "system:",
            "human:",
            "[inst]",
            "<|im_start|>",
            "###instruction",
        ],
        JailbreakSignal.TOKEN_SMUGGLING: [
            "b64decode",
            "rot13",
            "base64",
            "hex decode",
            "unicode escape",
        ],
    }

    def detect(self, text: str) -> JailbreakDetection:
        lowered = text.lower()
        matched_signals: list[JailbreakSignal] = []

        for signal, patterns in self.SIGNAL_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in lowered:
                    matched_signals.append(signal)
                    break

        confidence = len(matched_signals) / len(JailbreakSignal)
        is_jailbreak = confidence >= 0.2
        explanation = f"Detected signals: {[s.value for s in matched_signals]}"

        return JailbreakDetection(
            signals=matched_signals,
            confidence=confidence,
            is_jailbreak=is_jailbreak,
            explanation=explanation,
        )

    def batch_detect(self, texts: list[str]) -> list[JailbreakDetection]:
        return [self.detect(t) for t in texts]
