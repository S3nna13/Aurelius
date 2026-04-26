"""Reference-based evaluation metrics for model outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExactMatchScorer:
    """Binary exact match scorer for model outputs."""

    case_sensitive: bool = False

    def score(self, reference: str, candidate: str) -> float:
        """Return 1.0 if exact match, 0.0 otherwise."""
        ref = reference if self.case_sensitive else reference.lower()
        cand = candidate if self.case_sensitive else candidate.lower()
        return 1.0 if ref == cand else 0.0

    def score_batch(self, references: list[str], candidates: list[str]) -> list[float]:
        return [self.score(r, c) for r, c in zip(references, candidates)]


EXACT_MATCH = ExactMatchScorer()
