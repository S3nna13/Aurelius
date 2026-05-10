from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    category: str
    confidence: float
    requires_retrieval: bool = False
    requires_tools: bool = False
    requires_reasoning: bool = False
    requires_multimodal: bool = False
    privacy_sensitive: bool = False
    reasoning_depth: str = "low"  # low | medium | high
    context_length_hint: int = 4096
    modalities: list[str] = field(default_factory=lambda: ["text"])


_INTENT_PATTERNS: list[tuple[str, str, dict[str, Any]]] = [
    ("greeting", r"\b(hi|hello|hey|greetings)\b", {}),
    ("faq", r"\b(what is|how do|how to|what does|define|explain)\b", {}),
    (
        "research",
        r"\b(analyze|research|investigate|compare|evaluate|summarize|synthesize)\b",
        {"requires_retrieval": True, "requires_reasoning": True, "reasoning_depth": "high"},
    ),
    (
        "code",
        r"\b(code|program|function|debug|refactor|implement|write a|fix bug|algorithm)\b",
        {"requires_tools": True, "requires_reasoning": True, "reasoning_depth": "medium"},
    ),
    (
        "math",
        r"\b(calculate|compute|solve|equation|derivative|integral|matrix)\b",
        {"requires_tools": True, "requires_reasoning": True, "reasoning_depth": "high"},
    ),
    (
        "analysis",
        r"\b(why|how come|what caused|implications|pros and cons|tradeoff)\b",
        {"requires_reasoning": True, "reasoning_depth": "medium"},
    ),
    (
        "document",
        r"\b(read|document|pdf|paper|article|report|long text|chapter)\b",
        {"requires_retrieval": True, "context_length_hint": 32768},
    ),
    (
        "creative",
        r"\b(write|draft|compose|create|design|generate|story|poem)\b",
        {"reasoning_depth": "low"},
    ),
    (
        "multimodal",
        r"\b(image|picture|photo|diagram|chart|graph|video|audio)\b",
        {"requires_multimodal": True, "modalities": ["text", "image"]},
    ),
    (
        "sensitive",
        r"\b(confidential|private|secret|password|credential|pii|personal|health|financial)\b",
        {"privacy_sensitive": True},
    ),
    (
        "translation",
        r"\b(translate|translate to|in spanish|in french|in german)\b",
        {},
    ),
    (
        "tool_use",
        r"\b(search|lookup|find|fetch|retrieve|query|run|execute|call api)\b",
        {"requires_tools": True, "requires_retrieval": True},
    ),
]


class IntentClassifier:
    """Classify user intent based on message content patterns.

    Uses regex patterns for fast, lightweight classification before
    any model invocation. Supports custom classifiers via the
    ``classifiers`` registry (callable injection pattern).
    """

    def __init__(
        self,
        patterns: list[tuple[str, str, dict[str, Any]]] | None = None,
        classifiers: dict[str, Callable[[str], float]] | None = None,
    ) -> None:
        self._patterns = patterns or _INTENT_PATTERNS
        self._classifiers: dict[str, Callable[[str], float]] = classifiers or {}

    def classify(self, message: str) -> Intent:
        """Classify a user message and return the best-matching intent."""
        message_lower = message.lower()

        best_category = "general"
        best_confidence = 0.0
        best_props: dict[str, Any] = {}

        for category, pattern, props in self._patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                confidence = 0.7
                # Prefer more specific patterns by length
                confidence += min(len(pattern) / 100.0, 0.25)
                if confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence
                    best_props = props

        # Run custom classifiers (e.g., embedding-based semantic classification)
        for name, classifier_fn in self._classifiers.items():
            score = classifier_fn(message)
            if score > best_confidence:
                best_category = name
                best_confidence = score
                best_props = {}

        return Intent(
            category=best_category,
            confidence=best_confidence,
            requires_retrieval=best_props.get("requires_retrieval", False),
            requires_tools=best_props.get("requires_tools", False),
            requires_reasoning=best_props.get("requires_reasoning", False),
            requires_multimodal=best_props.get("requires_multimodal", False),
            privacy_sensitive=best_props.get("privacy_sensitive", False),
            reasoning_depth=best_props.get("reasoning_depth", "low"),
            context_length_hint=best_props.get("context_length_hint", 4096),
            modalities=best_props.get("modalities", ["text"]),
        )
