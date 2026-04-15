"""Curriculum learning: present training examples in order of increasing difficulty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    """Configuration for a curriculum learning schedule."""
    strategy: str = "linear"          # "linear", "exponential", or "step"
    n_stages: int = 5
    warmup_steps: int = 100
    total_steps: int = 1000
    difficulty_metric: str = "loss"   # "loss" or "perplexity"


@dataclass
class DifficultyScore:
    """Stores a difficulty score for a single training sample."""
    sample_id: int
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Curriculum weight functions
# ---------------------------------------------------------------------------

def linear_curriculum_weight(step: int, total_steps: int) -> float:
    """Return a value in [0, 1] growing linearly from 0 to 1 over total_steps."""
    if total_steps <= 0:
        return 1.0
    return float(min(max(step / total_steps, 0.0), 1.0))


def exponential_curriculum_weight(
    step: int, total_steps: int, k: float = 5.0
) -> float:
    """Exponential growth: (exp(k * step/total_steps) - 1) / (exp(k) - 1), clamped to [0, 1]."""
    if total_steps <= 0:
        return 1.0
    t = float(step) / float(total_steps)
    t = min(max(t, 0.0), 1.0)
    denom = math.exp(k) - 1.0
    if denom == 0.0:
        return t
    weight = (math.exp(k * t) - 1.0) / denom
    return float(min(max(weight, 0.0), 1.0))


def step_curriculum_weight(step: int, n_stages: int, total_steps: int) -> float:
    """Staircase schedule: divide total_steps into n_stages equal intervals.

    Returns stage / n_stages where stage is the current stage index (0-based).
    """
    if n_stages <= 0 or total_steps <= 0:
        return 1.0
    step = min(max(step, 0), total_steps)
    # Which stage are we in? (0-indexed)
    stage = int(step * n_stages / total_steps)
    # Clamp to [0, n_stages - 1] so we never exceed 1 before total_steps
    stage = min(stage, n_stages - 1)
    return float(stage) / float(n_stages)


# ---------------------------------------------------------------------------
# DifficultyRanker
# ---------------------------------------------------------------------------

class DifficultyRanker:
    """Ranks samples by their difficulty scores."""

    def __init__(self, scores: List[DifficultyScore]) -> None:
        self._scores: List[DifficultyScore] = list(scores)

    def rank(self, ascending: bool = True) -> List[DifficultyScore]:
        """Return a copy of scores sorted by score value."""
        return sorted(self._scores, key=lambda ds: ds.score, reverse=not ascending)

    def get_percentile(self, pct: float) -> float:
        """Return the score at the given percentile (0–100) using linear interpolation."""
        if not self._scores:
            raise ValueError("No scores available.")
        sorted_scores = [ds.score for ds in self.rank(ascending=True)]
        n = len(sorted_scores)
        if n == 1:
            return sorted_scores[0]
        # Map pct in [0, 100] to index space [0, n-1]
        idx = pct / 100.0 * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        lo = max(0, min(lo, n - 1))
        hi = max(0, min(hi, n - 1))
        if lo == hi:
            return sorted_scores[lo]
        frac = idx - lo
        return sorted_scores[lo] + frac * (sorted_scores[hi] - sorted_scores[lo])

    def get_easy_samples(self, fraction: float) -> List[DifficultyScore]:
        """Return the easiest (lowest score) fraction of samples."""
        ranked = self.rank(ascending=True)
        n = max(1, int(math.ceil(len(ranked) * fraction)))
        return ranked[:n]

    def get_hard_samples(self, fraction: float) -> List[DifficultyScore]:
        """Return the hardest (highest score) fraction of samples."""
        ranked = self.rank(ascending=False)
        n = max(1, int(math.ceil(len(ranked) * fraction)))
        return ranked[:n]


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """Samples from the dataset according to a curriculum schedule."""

    def __init__(self, ranker: DifficultyRanker, config: CurriculumConfig) -> None:
        self._ranker = ranker
        self._config = config
        self._dirty = False  # tracks whether scores were updated since last rank

    def get_curriculum_fraction(self, step: int) -> float:
        """Return the fraction of the dataset (easy-to-hard) to sample from at this step."""
        cfg = self._config
        if cfg.strategy == "linear":
            weight = linear_curriculum_weight(step, cfg.total_steps)
        elif cfg.strategy == "exponential":
            weight = exponential_curriculum_weight(step, cfg.total_steps)
        elif cfg.strategy == "step":
            weight = step_curriculum_weight(step, cfg.n_stages, cfg.total_steps)
        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy!r}")
        # Always expose at least a small fraction so we can sample
        return float(max(weight, 1.0 / max(len(self._ranker._scores), 1)))

    def sample_indices(self, step: int, n_samples: int) -> List[int]:
        """Return n_samples sample_ids from the curriculum-appropriate difficulty range."""
        fraction = self.get_curriculum_fraction(step)
        easy_pool = self._ranker.get_easy_samples(fraction)
        if not easy_pool:
            easy_pool = self._ranker.rank(ascending=True)

        # Cycle through the pool to fill n_samples
        ids = [ds.sample_id for ds in easy_pool]
        result: List[int] = []
        pool_size = len(ids)
        for i in range(n_samples):
            result.append(ids[i % pool_size])
        return result

    def update_scores(self, updates: List[DifficultyScore]) -> None:
        """Update scores for given sample_ids; re-ranking happens on next call."""
        update_map = {ds.sample_id: ds for ds in updates}
        new_scores: List[DifficultyScore] = []
        existing_ids = {ds.sample_id for ds in self._ranker._scores}
        for ds in self._ranker._scores:
            if ds.sample_id in update_map:
                new_scores.append(update_map[ds.sample_id])
            else:
                new_scores.append(ds)
        # Add brand-new sample_ids not previously seen
        for sid, ds in update_map.items():
            if sid not in existing_ids:
                new_scores.append(ds)
        self._ranker._scores = new_scores
        self._dirty = True


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_sample_difficulty(losses: torch.Tensor) -> List[float]:
    """Convert per-sample losses (1-D tensor) to difficulty scores (list of floats)."""
    return losses.detach().float().tolist()
