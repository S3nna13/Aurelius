"""Aurelius perplexity scorer: computes perplexity for language model evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PerplexityResult:
    text: str
    log_prob_sum: float
    token_count: int
    perplexity: float


class PerplexityScorer:
    """Computes perplexity from log-probabilities."""

    def __init__(self, base: float = math.e) -> None:
        self.base = base

    def from_log_probs(self, text: str, log_probs: list[float]) -> PerplexityResult:
        """Compute perplexity = exp(-mean(log_probs)).

        Empty log_probs yields perplexity = float("inf").
        """
        if not log_probs:
            return PerplexityResult(
                text=text,
                log_prob_sum=0.0,
                token_count=0,
                perplexity=float("inf"),
            )

        log_prob_sum = sum(log_probs)
        token_count = len(log_probs)
        mean_log_prob = log_prob_sum / token_count
        perplexity = math.exp(-mean_log_prob)
        return PerplexityResult(
            text=text,
            log_prob_sum=log_prob_sum,
            token_count=token_count,
            perplexity=perplexity,
        )

    def compare(self, a: PerplexityResult, b: PerplexityResult) -> str:
        """Return 'a' if a has lower perplexity, 'b' if b is lower, 'tie' if equal."""
        if a.perplexity < b.perplexity:
            return "a"
        if b.perplexity < a.perplexity:
            return "b"
        return "tie"

    def batch_score(
        self, examples: list[tuple[str, list[float]]]
    ) -> list[PerplexityResult]:
        """Run from_log_probs for each (text, log_probs) pair."""
        return [self.from_log_probs(text, log_probs) for text, log_probs in examples]

    def summary(self, results: list[PerplexityResult]) -> dict:
        """Return aggregate statistics over a list of PerplexityResult objects."""
        count = len(results)
        if count == 0:
            return {
                "count": 0,
                "mean_perplexity": 0.0,
                "min_perplexity": 0.0,
                "max_perplexity": 0.0,
            }
        perplexities = [r.perplexity for r in results]
        return {
            "count": count,
            "mean_perplexity": sum(perplexities) / count,
            "min_perplexity": min(perplexities),
            "max_perplexity": max(perplexities),
        }


PERPLEXITY_SCORER_REGISTRY = {"default": PerplexityScorer}
