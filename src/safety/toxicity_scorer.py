"""Toxicity Scorer — keyword/pattern-based toxicity scoring for Aurelius.

Scores text across six harm categories using pattern matching.
No external ML frameworks required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------


class ToxicityCategory(StrEnum):
    HATE = "hate"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    SPAM = "spam"


@dataclass
class ToxicityScore:
    text: str
    scores: dict[ToxicityCategory, float]
    overall: float
    flagged: bool


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ToxicityScorer:
    """Heuristic toxicity scorer using pattern matching per category."""

    PATTERNS: dict[ToxicityCategory, list[str]] = {
        ToxicityCategory.HATE: [
            r"\b(hate|despise)\s+(all\s+)?(those\s+)?(people|them|you)\b",
            r"\b(subhuman|inferior race|white supremac|racial slur)\b",
            r"\b(ethnic\s+cleansing|genocide\s+of)\b",
            r"\b(go back to your country|you don't belong here)\b",
            r"\b(death to all|exterminate the)\b",
        ],
        ToxicityCategory.VIOLENCE: [
            r"\b(kill|murder|assassinate|slaughter)\s+(him|her|them|you|everyone)\b",
            r"\b(stab|shoot|bomb|blow up)\s+(the|a|him|her|them)?\s*\b",
            r"\b(beat (him|her|them|you) (up|to death))\b",
            r"\b(torture|dismember|decapitate)\b",
            r"\b(mass shooting|terrorist attack|violent attack)\b",
        ],
        ToxicityCategory.SEXUAL: [
            r"\b(explicit\s+sexual|pornographic|nude\s+image)\b",
            r"\b(child\s+(pornography|sexual|exploitation))\b",
            r"\b(sexual\s+(assault|abuse|harassment|coercion))\b",
            r"\b(rape|molest|grope)\b",
            r"\b(underage\s+sex|minor\s+(sex|nude))\b",
        ],
        ToxicityCategory.SELF_HARM: [
            r"\b(kill\s+myself|end\s+my\s+life|commit\s+suicide)\b",
            r"\b(cut\s+myself|self[\s-]harm|self[\s-]injur)\b",
            r"\b(overdose\s+on|take\s+too\s+many\s+pills)\b",
            r"\b(jump\s+off\s+(a\s+bridge|building|cliff))\b",
            r"\b(want\s+to\s+die|no\s+reason\s+to\s+live)\b",
        ],
        ToxicityCategory.HARASSMENT: [
            r"\b(you\s+are\s+(worthless|pathetic|disgusting|a\s+loser))\b",
            r"\b(nobody\s+likes\s+you|everyone\s+hates\s+you)\b",
            r"\b(i\s+will\s+(ruin|destroy|expose)\s+you)\b",
            r"\b(go\s+(die|kill\s+yourself|away\s+forever))\b",
            r"\b(doxx|swat\s+you|leak\s+your\s+address)\b",
        ],
        ToxicityCategory.SPAM: [
            r"\b(click\s+here\s+to\s+(win|claim|get\s+free))\b",
            r"\b(you\s+have\s+(won|been\s+selected)\s+(a\s+prize|lottery))\b",
            r"\b(buy\s+now|limited\s+time\s+offer|act\s+fast)\b",
            r"\b(send\s+me\s+your\s+(bank|credit\s+card|ssn|password))\b",
            r"\b(nigerian\s+prince|wire\s+transfer\s+(fee|advance))\b",
        ],
    }

    def score(self, text: str, threshold: float = 0.5) -> ToxicityScore:
        """Score a single text string for toxicity across all categories.

        Pattern matches are counted per category, then normalized by text
        length (in words, minimum 1) to produce a [0, 1] score.

        Args:
            text: Input text to score.
            threshold: overall >= threshold sets flagged=True.

        Returns:
            ToxicityScore with per-category scores, overall, and flagged flag.
        """
        lower = text.lower()
        word_count = max(len(text.split()), 1)
        category_scores: dict[ToxicityCategory, float] = {}

        for category, patterns in self.PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, lower)
                match_count += len(matches)
            # Normalize: clamp to [0, 1] — more than one match per 10 words = 1.0
            raw = match_count / (word_count / 10.0)
            category_scores[category] = min(raw, 1.0)

        overall = max(category_scores.values()) if category_scores else 0.0
        flagged = overall >= threshold

        return ToxicityScore(
            text=text,
            scores=category_scores,
            overall=overall,
            flagged=flagged,
        )

    def batch_score(self, texts: list[str], threshold: float = 0.5) -> list[ToxicityScore]:
        """Score a list of texts.

        Args:
            texts: List of input strings.
            threshold: Passed to each score() call.

        Returns:
            List of ToxicityScore in input order.
        """
        return [self.score(t, threshold=threshold) for t in texts]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAFETY_REGISTRY: dict = {}
SAFETY_REGISTRY["toxicity_scorer"] = ToxicityScorer()
