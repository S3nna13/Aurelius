"""Aurelius faithfulness scorer: lexical-overlap faithfulness scoring (NLI-free)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    supported_claims: int
    total_claims: int
    unsupported: list[str]


class FaithfulnessScorer:
    """Scores how faithful a generated answer is to a source document
    using Jaccard word overlap — no external models required.
    """

    def _split_claims(self, text: str) -> list[str]:
        """Split text into individual claims on sentence-ending punctuation."""
        # Split on '. ', '? ', or '! ' (sentence-ending sequences)
        parts = re.split(r"(?<=[.?!])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _word_overlap(self, claim: str, source: str) -> float:
        """Jaccard similarity on word sets between claim and source.

        Returns 0.0 if both are empty.
        """
        claim_words = set(re.findall(r"\w+", claim.lower()))
        source_words = set(re.findall(r"\w+", source.lower()))

        if not claim_words and not source_words:
            return 0.0

        intersection = claim_words & source_words
        union = claim_words | source_words

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def score(self, generated: str, source: str, threshold: float = 0.3) -> FaithfulnessResult:
        """Score faithfulness of generated text against source.

        A claim is supported if its word overlap with source >= threshold.
        Returns score=1.0 when there are no claims (empty generated text).
        """
        claims = self._split_claims(generated)
        total = len(claims)

        if total == 0:
            return FaithfulnessResult(
                score=1.0,
                supported_claims=0,
                total_claims=0,
                unsupported=[],
            )

        supported = 0
        unsupported: list[str] = []
        for claim in claims:
            overlap = self._word_overlap(claim, source)
            if overlap >= threshold:
                supported += 1
            else:
                unsupported.append(claim)

        faithfulness_score = supported / total
        return FaithfulnessResult(
            score=faithfulness_score,
            supported_claims=supported,
            total_claims=total,
            unsupported=unsupported,
        )

    def batch_score(
        self,
        pairs: list[tuple[str, str]],
        threshold: float = 0.3,
    ) -> list[FaithfulnessResult]:
        """Score faithfulness for a list of (generated, source) pairs."""
        return [self.score(generated, source, threshold) for generated, source in pairs]


FAITHFULNESS_SCORER_REGISTRY = {"default": FaithfulnessScorer}
