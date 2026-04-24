"""Content classifier — Aurelius safety module.

Keyword-based content safety classifier. Stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentCategory(Enum):
    SAFE = "safe"
    VIOLENCE = "violence"
    HATE = "hate"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    ILLEGAL = "illegal"
    SPAM = "spam"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a single text."""
    text: str
    category: ContentCategory
    confidence: float
    evidence: List[str]


# ---------------------------------------------------------------------------
# Keyword lists per category
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[ContentCategory, List[str]] = {
    ContentCategory.VIOLENCE: [
        "kill", "murder", "weapon", "bomb", "attack", "shoot", "stab",
    ],
    ContentCategory.HATE: [
        "slur", "racist", "bigot", "supremacist", "dehumanize",
    ],
    ContentCategory.SEXUAL: [
        "explicit", "nude", "pornographic", "sexual content",
    ],
    ContentCategory.SELF_HARM: [
        "suicide", "self-harm", "cutting", "overdose",
    ],
    ContentCategory.ILLEGAL: [
        "cocaine", "heroin", "counterfeit", "launder", "trafficking",
    ],
    ContentCategory.SPAM: [
        "click here", "win money", "free offer", "limited time", "act now",
    ],
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ContentClassifier:
    """Keyword-based content safety classifier."""

    def __init__(self) -> None:
        # Store as-is; matching is done against lowercased text
        self._keywords: dict[ContentCategory, List[str]] = {
            cat: list(kws) for cat, kws in _CATEGORY_KEYWORDS.items()
        }

    def classify(self, text: str) -> ClassificationResult:
        """Classify a single text string.

        Lowercases the text, counts keyword matches per category, selects
        the category with the most matches. Confidence = min(matches/3, 1.0).
        If no matches, returns SAFE with confidence=1.0.
        Evidence is the sorted unique list of matched keywords.
        """
        lower = text.lower()
        best_category: ContentCategory = ContentCategory.SAFE
        best_count: int = 0
        all_evidence: set[str] = set()
        category_counts: dict[ContentCategory, int] = {}

        for category, keywords in self._keywords.items():
            hits: list[str] = []
            for kw in keywords:
                if kw in lower:
                    hits.append(kw)
            count = len(hits)
            category_counts[category] = count
            if count > 0:
                all_evidence.update(hits)
            if count > best_count:
                best_count = count
                best_category = category

        if best_count == 0:
            return ClassificationResult(
                text=text,
                category=ContentCategory.SAFE,
                confidence=1.0,
                evidence=[],
            )

        confidence = min(best_count / 3.0, 1.0)
        evidence = sorted(all_evidence)

        return ClassificationResult(
            text=text,
            category=best_category,
            confidence=confidence,
            evidence=evidence,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify a list of texts, returning one result per text."""
        return [self.classify(t) for t in texts]

    def safe_to_serve(
        self, result: ClassificationResult, threshold: float = 0.5
    ) -> bool:
        """Return True if the result is safe to serve.

        A result is safe to serve if:
        - Its category is SAFE, OR
        - Its confidence is strictly below the threshold.
        """
        return result.category == ContentCategory.SAFE or result.confidence < threshold


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONTENT_CLASSIFIER_REGISTRY: dict[str, type] = {
    "default": ContentClassifier,
}
