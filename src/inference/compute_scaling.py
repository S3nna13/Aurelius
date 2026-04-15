"""Inference-time compute scaling (Best-of-N, Majority Vote, Pass@k).

Trades more compute at inference time for better outputs by generating
N independent responses and selecting the best via a verifier or vote.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class ScalingConfig:
    """Configuration for inference-time compute scaling."""

    n_samples: int = 8
    temperature: float = 0.8
    aggregation: str = "majority_vote"  # "majority_vote" | "best_of_n" | "weighted_majority"


# ---------------------------------------------------------------------------
# Standalone aggregation helpers
# ---------------------------------------------------------------------------

def majority_vote(responses: List[str]) -> str:
    """Return the most common response; tie-break by first occurrence."""
    if not responses:
        return ""

    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for idx, r in enumerate(responses):
        counts[r] = counts.get(r, 0) + 1
        if r not in first_seen:
            first_seen[r] = idx

    max_count = max(counts.values())
    # All candidates with the highest count
    candidates = [r for r, c in counts.items() if c == max_count]
    # Tie-break: first occurrence in the original list
    candidates.sort(key=lambda r: first_seen[r])
    return candidates[0]


def weighted_majority_vote(responses: List[str], weights: List[float]) -> str:
    """Sum weights per unique response, return the highest-weight one.

    Tie-break by first occurrence.
    """
    if not responses:
        return ""

    totals: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for idx, (r, w) in enumerate(zip(responses, weights)):
        totals[r] = totals.get(r, 0.0) + w
        if r not in first_seen:
            first_seen[r] = idx

    max_weight = max(totals.values())
    candidates = [r for r, w in totals.items() if w == max_weight]
    candidates.sort(key=lambda r: first_seen[r])
    return candidates[0]


def best_of_n(responses: List[str], scores: List[float]) -> str:
    """Return the response with the highest score."""
    if not responses:
        return ""
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return responses[best_idx]


def normalize_scores(scores: List[float]) -> List[float]:
    """Softmax-normalize scores so they sum to 1."""
    if not scores:
        return []
    t = torch.tensor(scores, dtype=torch.float32)
    normalized = torch.softmax(t, dim=0)
    return normalized.tolist()


def compute_response_diversity(responses: List[str]) -> float:
    """Diversity of responses: 1.0 = all unique, 0.0 = all the same.

    Computed as (n_unique - 1) / (n - 1) for n > 1.
    Returns 0.0 for empty list or single-element list.
    Uses exact string match.
    """
    n = len(responses)
    if n <= 1:
        return 0.0
    n_unique = len(set(responses))
    return (n_unique - 1) / (n - 1)


# ---------------------------------------------------------------------------
# ComputeScaler
# ---------------------------------------------------------------------------

class ComputeScaler:
    """Generate N responses and aggregate them for inference-time scaling.

    Args:
        generate_fn: Callable[[str], str] — takes a prompt, returns a response.
        score_fn: Optional[Callable[[str, str], float]] — takes (prompt, response)
                  and returns a float score. Required for "best_of_n" aggregation.
        config: ScalingConfig (defaults to ScalingConfig() if None).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        score_fn: Optional[Callable[[str, str], float]] = None,
        config: Optional[ScalingConfig] = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.score_fn = score_fn
        self.config = config if config is not None else ScalingConfig()

    def generate_n(self, prompt: str) -> List[str]:
        """Call generate_fn n_samples times and return the list of responses."""
        return [self.generate_fn(prompt) for _ in range(self.config.n_samples)]

    def aggregate(self, prompt: str, responses: List[str]) -> str:
        """Apply the configured aggregation strategy to select the best response."""
        agg = self.config.aggregation
        if agg == "majority_vote":
            return majority_vote(responses)
        elif agg == "best_of_n":
            if self.score_fn is None:
                raise ValueError("score_fn is required for 'best_of_n' aggregation")
            scores = [self.score_fn(prompt, r) for r in responses]
            return best_of_n(responses, scores)
        elif agg == "weighted_majority":
            if self.score_fn is None:
                # Fall back to uniform weights
                weights = [1.0] * len(responses)
            else:
                weights = [self.score_fn(prompt, r) for r in responses]
            return weighted_majority_vote(responses, weights)
        else:
            raise ValueError(f"Unknown aggregation: {agg!r}")

    def __call__(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Generate N responses, aggregate, and return (best_response, metadata).

        Metadata contains:
          - n_samples: number of generated responses
          - n_unique: number of unique responses
          - diversity: fraction of unique responses
          - aggregation: the aggregation method used
        """
        responses = self.generate_n(prompt)
        best = self.aggregate(prompt, responses)
        n_unique = len(set(responses))
        diversity = compute_response_diversity(responses)
        metadata: Dict[str, Any] = {
            "n_samples": self.config.n_samples,
            "n_unique": n_unique,
            "diversity": diversity,
            "aggregation": self.config.aggregation,
        }
        return best, metadata


# ---------------------------------------------------------------------------
# PassAtK
# ---------------------------------------------------------------------------

class PassAtK:
    """Estimate pass@k for code generation evaluation.

    Based on Chen et al. (2021): pass@k = 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = correct samples, k = number of attempts reported.
    """

    def estimate(self, n_samples: int, k: int, c: int) -> float:
        """Estimate pass@k analytically.

        Args:
            n_samples: Total number of generated solutions.
            k: Number of allowed attempts.
            c: Number of correct solutions among n_samples.

        Returns:
            Probability of passing at k attempts.
        """
        if c == 0:
            return 0.0
        if n_samples - c < k:
            return 1.0
        # 1 - C(n-c, k) / C(n, k)
        numerator = math.comb(n_samples - c, k)
        denominator = math.comb(n_samples, k)
        if denominator == 0:
            return 0.0
        return 1.0 - numerator / denominator

    def compute_from_results(self, results: List[bool], k: int) -> float:
        """Compute empirical pass@k from a list of pass/fail booleans.

        Args:
            results: List of booleans (True = correct solution).
            k: Number of allowed attempts.

        Returns:
            Empirical pass@k estimate.
        """
        n = len(results)
        c = sum(results)
        return self.estimate(n, k, c)
