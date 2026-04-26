from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Intent(StrEnum):
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    CONFIRMATION = "confirmation"
    DENIAL = "denial"
    GREETING = "greeting"
    FAREWELL = "farewell"
    CLARIFICATION_REQUEST = "clarification_request"
    INFORMATION_PROVISION = "information_provision"
    UNKNOWN = "unknown"


_DEFAULT_PATTERNS: dict[Intent, list[str]] = {
    Intent.QUESTION: [
        r"\b(what|who|where|when|why|how|which)\b",
        r"\?$",
    ],
    Intent.REQUEST: [
        r"\bplease\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"\bi want\b",
        r"\bi need\b",
        r"\bgive me\b",
    ],
    Intent.COMPLAINT: [
        r"\bnot working\b",
        r"\bbroken\b",
        r"\bwrong\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\berror\b",
    ],
    Intent.CONFIRMATION: [
        r"\byes\b",
        r"\bcorrect\b",
        r"\bexactly\b",
        r"\bright\b",
        r"\bsure\b",
        r"\bokay\b",
        r"\bok\b",
    ],
    Intent.DENIAL: [
        r"\bno\b",
        r"\bnot\b",
        r"\bwrong\b",
        r"\bnever\b",
        r"\bdon't\b",
        r"\bdont\b",
    ],
    Intent.GREETING: [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bgood morning\b",
        r"\bgood afternoon\b",
        r"\bgood evening\b",
    ],
    Intent.FAREWELL: [
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bsee you\b",
        r"\bthanks\b",
        r"\bthank you\b",
    ],
    Intent.CLARIFICATION_REQUEST: [
        r"what do you mean",
        r"could you explain",
        r"\bclarify\b",
    ],
    Intent.INFORMATION_PROVISION: [
        r"\bmy\b",
        r"\bi am\b",
        r"\bi'm\b",
        r"\bthe\b.{1,30}\bis\b",
    ],
}

_PRIORITY_ORDER: list[Intent] = [
    Intent.CLARIFICATION_REQUEST,
    Intent.FAREWELL,
    Intent.GREETING,
    Intent.COMPLAINT,
    Intent.CONFIRMATION,
    Intent.DENIAL,
    Intent.REQUEST,
    Intent.QUESTION,
    Intent.INFORMATION_PROVISION,
    Intent.UNKNOWN,
]

_NUMBER_RE = re.compile(r"\d+\.?\d*")
_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?")
_QUOTED_RE = re.compile(r'"([^"]+)"')


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    entities: dict
    keywords_matched: list[str] = field(default_factory=list)


class IntentClassifier:
    """Rule-based intent classifier with entity extraction."""

    def __init__(
        self,
        custom_patterns: dict[Intent, list[str]] | None = None,
    ) -> None:
        self._patterns: dict[Intent, list[re.Pattern]] = {}
        for intent, pats in _DEFAULT_PATTERNS.items():
            self._patterns[intent] = [re.compile(p, re.IGNORECASE) for p in pats]
        if custom_patterns:
            for intent, pats in custom_patterns.items():
                existing = self._patterns.get(intent, [])
                existing.extend(re.compile(p, re.IGNORECASE) for p in pats)
                self._patterns[intent] = existing

    def classify(self, text: str) -> IntentResult:
        scores: dict[Intent, float] = {i: 0.0 for i in Intent}
        matched: dict[Intent, list[str]] = {i: [] for i in Intent}

        for intent, compiled in self._patterns.items():
            for pat in compiled:
                m = pat.search(text)
                if m:
                    hit = m.group(0)
                    weight = 2.0 if " " in hit else 1.0
                    scores[intent] += weight
                    matched[intent].append(hit)

        top_score = max(scores.values())
        if top_score == 0.0:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                entities=self.extract_entities(text),
                keywords_matched=[],
            )

        candidates = {i for i, s in scores.items() if s == top_score}
        best = next(
            (i for i in _PRIORITY_ORDER if i in candidates),
            Intent.UNKNOWN,
        )

        total = sum(scores.values())
        confidence = round(scores[best] / total, 4) if total > 0 else 0.0
        return IntentResult(
            intent=best,
            confidence=min(confidence, 1.0),
            entities=self.extract_entities(text),
            keywords_matched=matched[best],
        )

    def batch_classify(self, texts: list[str]) -> list[IntentResult]:
        return [self.classify(t) for t in texts]

    def add_pattern(self, intent: Intent, pattern: str) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)
        self._patterns.setdefault(intent, []).append(compiled)

    def extract_entities(self, text: str) -> dict:
        entities: dict[str, list] = {}

        dates = _DATE_RE.findall(text)
        if dates:
            entities["dates"] = dates

        remaining = _DATE_RE.sub("", text)
        numbers = _NUMBER_RE.findall(remaining)
        if numbers:
            entities["numbers"] = numbers

        quoted = _QUOTED_RE.findall(text)
        if quoted:
            entities["quoted"] = quoted

        return entities
