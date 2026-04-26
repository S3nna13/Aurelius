"""
Pairwise preference data collection pipeline for DPO/RLHF training.

Extends basic preference collection with:
1. Best-of-N sampling to create diverse candidate pools
2. Automatic preference pair mining using reward model scores
3. Margin-based filtering (reject pairs with small reward gaps)
4. Diversity-aware sampling (avoid near-duplicate pairs)
5. Dataset statistics and quality metrics
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Response:
    """A single model response with optional reward score and metadata."""

    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """A chosen/rejected preference pair with margin information."""

    prompt: str
    chosen: Response
    rejected: Response
    margin: float = 0.0  # chosen.score - rejected.score
    source: str = "collected"


@dataclass
class CollectionConfig:
    """Configuration for the preference collection pipeline."""

    n_candidates: int = 4  # best-of-N
    min_margin: float = 0.1  # minimum score gap to accept pair
    max_pairs_per_prompt: int = 3  # max pairs generated per prompt
    dedup_threshold: float = 0.8  # text similarity threshold for dedup
    include_ties: bool = False  # include pairs with margin=0


# ---------------------------------------------------------------------------
# Text similarity utilities
# ---------------------------------------------------------------------------


def compute_text_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word sets. Returns float in [0, 1].

    Identical strings → 1.0; completely disjoint word sets → 0.0.
    Empty strings are treated as empty sets; two empty strings → 1.0.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def is_near_duplicate(
    response_a: str,
    response_b: str,
    threshold: float = 0.8,
) -> bool:
    """True if responses are too similar (likely near-duplicates).

    Uses Jaccard word similarity; returns True when similarity >= threshold.
    """
    return compute_text_similarity(response_a, response_b) >= threshold


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_with_reward_model(
    prompt: str,
    responses: list[str],
    reward_fn: Callable[[str, str], float],
) -> list[Response]:
    """Apply reward_fn to each (prompt, response) pair.

    Returns list of Response objects with scores populated.
    """
    scored: list[Response] = []
    for text in responses:
        score = reward_fn(prompt, text)
        scored.append(Response(text=text, score=score))
    return scored


# ---------------------------------------------------------------------------
# Preference pair mining
# ---------------------------------------------------------------------------


def mine_preference_pairs(
    prompt: str,
    responses: list[Response],
    config: CollectionConfig,
) -> list[PreferencePair]:
    """Given N candidate responses with scores, mine preference pairs.

    Steps:
    1. Sort by score descending.
    2. Create (higher_score, lower_score) pairs from all combinations.
    3. Filter by min_margin.
    4. Filter out near-duplicate pairs.
    5. Limit to max_pairs_per_prompt.

    Returns list of PreferencePair.
    """
    if not responses:
        return []

    # Only consider responses that have scores
    scored = [r for r in responses if r.score is not None]
    if len(scored) < 2:
        return []

    # Sort descending by score
    sorted_responses = sorted(scored, key=lambda r: r.score, reverse=True)

    pairs: list[PreferencePair] = []

    for i in range(len(sorted_responses)):
        for j in range(i + 1, len(sorted_responses)):
            higher = sorted_responses[i]
            lower = sorted_responses[j]

            margin = higher.score - lower.score

            # Filter by margin
            if margin < config.min_margin:
                continue

            # Filter ties unless explicitly included
            if not config.include_ties and margin == 0.0:
                continue

            # Filter near-duplicates
            if is_near_duplicate(higher.text, lower.text, config.dedup_threshold):
                continue

            pair = PreferencePair(
                prompt=prompt,
                chosen=higher,
                rejected=lower,
                margin=margin,
                source="collected",
            )
            pairs.append(pair)

            if len(pairs) >= config.max_pairs_per_prompt:
                return pairs

    return pairs


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Quality statistics for a preference dataset."""

    n_pairs: int
    n_prompts: int
    mean_margin: float
    std_margin: float
    min_margin: float
    max_margin: float
    source_distribution: dict[str, int]  # {source: count}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class PreferenceDatasetV2:
    """Manages a preference dataset with quality filtering and statistics."""

    def __init__(
        self,
        pairs: list[PreferencePair] | None = None,
        config: CollectionConfig = None,
    ) -> None:
        self._config = config if config is not None else CollectionConfig()
        self._pairs: list[PreferencePair] = []
        if pairs:
            for pair in pairs:
                self.add(pair)

    def add(self, pair: PreferencePair) -> bool:
        """Add pair if it passes quality filters. Returns True if added."""
        # Must meet min_margin requirement
        if pair.margin < self._config.min_margin:
            return False
        # Reject ties unless configured to include them
        if not self._config.include_ties and pair.margin == 0.0:
            return False
        self._pairs.append(pair)
        return True

    def add_batch(self, pairs: list[PreferencePair]) -> int:
        """Add multiple pairs. Returns count added."""
        return sum(1 for p in pairs if self.add(p))

    def filter_by_margin(self, min_margin: float) -> PreferenceDatasetV2:
        """Return new dataset with only high-margin pairs."""
        filtered_pairs = [p for p in self._pairs if p.margin >= min_margin]
        new_ds = PreferenceDatasetV2(config=self._config)
        # Bypass normal add filtering to preserve already-validated pairs
        new_ds._pairs = filtered_pairs
        return new_ds

    def get_stats(self) -> DatasetStats:
        """Compute dataset statistics."""
        n = len(self._pairs)
        if n == 0:
            return DatasetStats(
                n_pairs=0,
                n_prompts=0,
                mean_margin=0.0,
                std_margin=0.0,
                min_margin=0.0,
                max_margin=0.0,
                source_distribution={},
            )

        margins = [p.margin for p in self._pairs]
        mean_m = sum(margins) / n
        variance = sum((m - mean_m) ** 2 for m in margins) / n
        std_m = math.sqrt(variance)

        prompts = {p.prompt for p in self._pairs}
        source_dist: dict[str, int] = {}
        for p in self._pairs:
            source_dist[p.source] = source_dist.get(p.source, 0) + 1

        return DatasetStats(
            n_pairs=n,
            n_prompts=len(prompts),
            mean_margin=mean_m,
            std_margin=std_m,
            min_margin=min(margins),
            max_margin=max(margins),
            source_distribution=source_dist,
        )

    def to_list(self) -> list[PreferencePair]:
        """Return all pairs as a list."""
        return list(self._pairs)

    def to_dicts(self) -> list[dict[str, str]]:
        """Export as list of {'prompt', 'chosen', 'rejected'} dicts for DPO."""
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen.text,
                "rejected": p.rejected.text,
            }
            for p in self._pairs
        ]

    def sample(self, n: int, strategy: str = "random") -> list[PreferencePair]:
        """Sample n pairs.

        strategy:
          - "random"      : uniform random sample
          - "high_margin" : top-n by margin
          - "diverse"     : spread by margin (evenly spaced across sorted margins)
        """
        if n <= 0:
            return []
        if strategy == "high_margin":
            sorted_pairs = sorted(self._pairs, key=lambda p: p.margin, reverse=True)
            return sorted_pairs[:n]
        elif strategy == "diverse":
            if len(self._pairs) == 0:
                return []
            sorted_pairs = sorted(self._pairs, key=lambda p: p.margin)
            if n >= len(sorted_pairs):
                return list(sorted_pairs)
            # Evenly-spaced indices across the sorted list
            step = (len(sorted_pairs) - 1) / (n - 1) if n > 1 else 0
            indices = {round(i * step) for i in range(n)}
            return [sorted_pairs[i] for i in sorted(indices)]
        else:  # "random"
            pool = list(self._pairs)
            k = min(n, len(pool))
            return random.sample(pool, k)

    def __len__(self) -> int:
        return len(self._pairs)


# ---------------------------------------------------------------------------
# Full pipeline collector
# ---------------------------------------------------------------------------


class PreferenceCollectorV2:
    """Full pipeline: generate candidates → score → mine pairs → filter → store."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],  # prompt → response
        reward_fn: Callable[[str, str], float],  # (prompt, response) → score
        config: CollectionConfig = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._reward_fn = reward_fn
        self._config = config if config is not None else CollectionConfig()
        self._dataset = PreferenceDatasetV2(config=self._config)

    def collect(
        self,
        prompt: str,
        n_candidates: int | None = None,
    ) -> list[PreferencePair]:
        """Generate N candidates, score, mine pairs. Return new pairs."""
        k = n_candidates if n_candidates is not None else self._config.n_candidates

        # Generate candidate responses
        raw_responses = [self._generate_fn(prompt) for _ in range(k)]

        # Score with reward model
        scored_responses = score_with_reward_model(prompt, raw_responses, self._reward_fn)

        # Mine preference pairs
        new_pairs = mine_preference_pairs(prompt, scored_responses, self._config)

        # Add to accumulated dataset
        self._dataset.add_batch(new_pairs)

        return new_pairs

    def collect_batch(
        self,
        prompts: list[str],
        n_candidates: int | None = None,
    ) -> PreferenceDatasetV2:
        """Collect from all prompts. Return dataset with all collected pairs."""
        for prompt in prompts:
            self.collect(prompt, n_candidates=n_candidates)
        return self._dataset

    @property
    def dataset(self) -> PreferenceDatasetV2:
        """Access accumulated dataset."""
        return self._dataset
