"""Advanced dataset mixing and curriculum strategies for multi-source training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


@dataclass
class DataSource:
    name: str
    data: List[Dict]            # list of {"input_ids": ..., "labels": ...} or similar
    weight: float = 1.0         # initial sampling weight
    domain: str = "general"     # domain label for domain-aware mixing
    metadata: Dict = field(default_factory=dict)


@dataclass
class MixerConfig:
    strategy: str = "weighted"  # "weighted" | "proportional" | "curriculum" | "domain_aware"
    temperature: float = 1.0    # for softmax weighting (higher = more uniform)
    min_weight: float = 0.01    # minimum weight per source
    curriculum_steps: int = 1000
    domain_weights: Dict[str, float] = field(default_factory=dict)
    seed: int = 42


def normalize_weights(weights: List[float], temperature: float = 1.0) -> List[float]:
    """
    Normalize weights to sum to 1.0 with temperature scaling.
    Temperature > 1 → more uniform; temperature < 1 → sharper.
    Uses softmax(log(w) / temperature).
    """
    if not weights:
        return []
    # Clamp to avoid log(0)
    safe_weights = [max(w, 1e-12) for w in weights]
    log_w = [math.log(w) / temperature for w in safe_weights]
    # Numerically stable softmax: subtract max
    max_log = max(log_w)
    exp_w = [math.exp(lw - max_log) for lw in log_w]
    total = sum(exp_w)
    return [e / total for e in exp_w]


def compute_proportional_weights(
    source_sizes: List[int],
    alpha: float = 0.7,          # exponent for size weighting (1.0=proportional, 0.5=sqrt)
) -> List[float]:
    """
    Compute weights proportional to dataset_size^alpha.
    Larger datasets get more weight, but not fully proportional (avoid dominance).
    """
    if not source_sizes:
        return []
    powered = [s ** alpha for s in source_sizes]
    total = sum(powered)
    if total == 0:
        n = len(source_sizes)
        return [1.0 / n] * n
    return [p / total for p in powered]


def compute_domain_weights(
    sources: List[DataSource],
    domain_config: Dict[str, float],  # domain_name → target weight
) -> List[float]:
    """
    Compute per-source weights based on domain assignments.
    Sources in the same domain share their domain's total weight equally.
    """
    if not sources:
        return []

    # Count sources per domain
    domain_counts: Dict[str, int] = {}
    for src in sources:
        domain_counts[src.domain] = domain_counts.get(src.domain, 0) + 1

    # Assign per-source weight = domain_weight / count_in_domain
    # For domains not in domain_config, assign equal share of remaining weight
    configured_total = sum(domain_config.values())
    remaining = max(0.0, 1.0 - configured_total)
    unconfigured_domains = [d for d in domain_counts if d not in domain_config]
    n_unconfigured = len(unconfigured_domains)

    weights = []
    for src in sources:
        if src.domain in domain_config:
            w = domain_config[src.domain] / domain_counts[src.domain]
        else:
            # Split remaining weight equally among unconfigured domains, then divide by count
            share = (remaining / n_unconfigured) if n_unconfigured > 0 else 0.0
            w = share / domain_counts[src.domain]
        weights.append(w)

    # Normalize so they sum to 1
    total = sum(weights)
    if total == 0:
        n = len(weights)
        return [1.0 / n] * n
    return [w / total for w in weights]


class CurriculumScheduler:
    """
    Adjust mixing weights over training steps.
    Phase 1: high weight on easy/small domains
    Phase 2: shift toward harder/larger domains
    """

    def __init__(
        self,
        sources: List[DataSource],
        easy_domains: Optional[List[str]] = None,  # domains to prioritize early
        n_steps: int = 1000,
        warmup_fraction: float = 0.3,
    ):
        self.sources = sources
        self.easy_domains = set(easy_domains or [])
        self.n_steps = n_steps
        self.warmup_fraction = warmup_fraction
        self._warmup_steps = int(n_steps * warmup_fraction)

        # Precompute base weights: easy_domains boosted at step 0, faded by end
        n = len(sources)
        self._easy_indices = [i for i, s in enumerate(sources) if s.domain in self.easy_domains]
        self._hard_indices = [i for i, s in enumerate(sources) if s.domain not in self.easy_domains]

    def get_weights(self, step: int) -> List[float]:
        """Return current mixing weights at given step."""
        n = len(self.sources)
        if n == 0:
            return []

        # Progress from 0.0 (step=0) to 1.0 (step=n_steps)
        progress = min(step / max(self.n_steps, 1), 1.0)

        weights = []
        for i, src in enumerate(self.sources):
            base = src.weight
            if i in self._easy_indices:
                # Easy domains: boosted early (factor 2→1 over warmup, then 1→0.5 after)
                if step <= self._warmup_steps:
                    boost = 2.0 - (step / max(self._warmup_steps, 1)) if self._warmup_steps > 0 else 1.0
                else:
                    post = (step - self._warmup_steps) / max(self.n_steps - self._warmup_steps, 1)
                    boost = 1.0 - 0.5 * post
                weights.append(base * boost)
            else:
                # Hard domains: suppressed early, boosted late
                if step <= self._warmup_steps:
                    suppress = 0.5 + 0.5 * (step / max(self._warmup_steps, 1)) if self._warmup_steps > 0 else 1.0
                else:
                    post = (step - self._warmup_steps) / max(self.n_steps - self._warmup_steps, 1)
                    suppress = 1.0 + 0.5 * post
                weights.append(base * suppress)

        # Normalize
        total = sum(weights)
        if total == 0:
            return [1.0 / n] * n
        return [w / total for w in weights]

    def is_warmed_up(self, step: int) -> bool:
        """True after warmup_fraction * n_steps steps."""
        return step >= self._warmup_steps


class DatasetMixer:
    """
    Infinite iterator that mixes multiple data sources with configurable strategy.
    """

    def __init__(
        self,
        sources: List[DataSource],
        config: MixerConfig = None,
    ):
        self.sources = sources
        self.config = config or MixerConfig()
        self._rng = random.Random(self.config.seed)

        # Per-source indices for cycling
        self._indices: List[List[int]] = []
        self._cursors: List[int] = []
        for src in sources:
            idx = list(range(len(src.data)))
            self._rng.shuffle(idx)
            self._indices.append(idx)
            self._cursors.append(0)

        # Weight history and stats
        self._weight_history: List[Dict[str, float]] = []
        self._stats: Dict[str, int] = {src.name: 0 for src in sources}

        # Curriculum scheduler (only for curriculum strategy)
        self._curriculum: Optional[CurriculumScheduler] = None
        if self.config.strategy == "curriculum":
            # Treat sources with lower index as "easy" if no domain specified
            easy_domains = list({s.domain for s in sources[:max(1, len(sources) // 2)]})
            self._curriculum = CurriculumScheduler(
                sources=sources,
                easy_domains=easy_domains,
                n_steps=self.config.curriculum_steps,
            )

        # Dynamic weights (mutable copy of source weights)
        self._weights: Dict[str, float] = {src.name: src.weight for src in sources}

    def _sample_source(self, weights: List[float]) -> int:
        """Weighted random selection of source index."""
        r = self._rng.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def _get_current_weights(self, step: int = 0) -> List[float]:
        """Get weights for current step (accounts for curriculum)."""
        strategy = self.config.strategy

        if strategy == "curriculum" and self._curriculum is not None:
            raw = self._curriculum.get_weights(step)
        elif strategy == "proportional":
            sizes = [len(src.data) for src in self.sources]
            raw = compute_proportional_weights(sizes)
        elif strategy == "domain_aware":
            raw = compute_domain_weights(self.sources, self.config.domain_weights)
        else:
            # "weighted" or default: use per-source weights with temperature
            raw_weights = [self._weights[src.name] for src in self.sources]
            raw = normalize_weights(raw_weights, self.config.temperature)

        # Enforce min_weight and renormalize
        min_w = self.config.min_weight
        clipped = [max(w, min_w) for w in raw]
        total = sum(clipped)
        return [w / total for w in clipped]

    def _next_example(self, source_idx: int) -> Dict:
        """Get the next example from a source, cycling when exhausted."""
        src = self.sources[source_idx]
        cursor = self._cursors[source_idx]
        idx_list = self._indices[source_idx]

        if cursor >= len(idx_list):
            # Reshuffle and reset
            self._rng.shuffle(idx_list)
            self._cursors[source_idx] = 0
            cursor = 0

        example = src.data[idx_list[cursor]]
        self._cursors[source_idx] = cursor + 1
        return example

    def __iter__(self) -> Iterator[Dict]:
        """Infinite iterator yielding examples from mixed sources."""
        step = 0
        while True:
            weights = self._get_current_weights(step)
            src_idx = self._sample_source(weights)
            example = self._next_example(src_idx)
            self._stats[self.sources[src_idx].name] += 1
            step += 1
            yield example

    def sample_batch(
        self,
        batch_size: int,
        step: int = 0,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Sample a batch of examples.
        Returns (examples, source_names) so caller knows which source each came from.
        """
        examples: List[Dict] = []
        source_names: List[str] = []

        weights = self._get_current_weights(step)
        for _ in range(batch_size):
            src_idx = self._sample_source(weights)
            example = self._next_example(src_idx)
            src_name = self.sources[src_idx].name
            self._stats[src_name] += 1
            examples.append(example)
            source_names.append(src_name)

        return examples, source_names

    def update_weights(self, source_name: str, new_weight: float) -> None:
        """Dynamically update weight for a named source."""
        if source_name not in self._weights:
            raise KeyError(f"Unknown source: {source_name!r}")
        old_weight = self._weights[source_name]
        self._weights[source_name] = new_weight
        # Record history
        self._weight_history.append({
            "source": source_name,
            "old_weight": old_weight,
            "new_weight": new_weight,
        })

    def get_weight_history(self) -> List[Dict[str, float]]:
        """Return history of weight adjustments."""
        return list(self._weight_history)

    def get_stats(self) -> Dict[str, Any]:
        """Return sampling statistics: {source_name: sample_count, ...}"""
        return dict(self._stats)


class AdaptiveMixer(DatasetMixer):
    """
    Mixer that adapts weights based on per-source training loss.
    Sources with higher loss → higher weight (focus on difficult domains).
    """

    def __init__(
        self,
        sources: List[DataSource],
        config: MixerConfig = None,
        adaptation_rate: float = 0.1,
    ):
        super().__init__(sources, config)
        self.adaptation_rate = adaptation_rate
        # Track EMA of losses per source
        self._ema_losses: Dict[str, float] = {src.name: 1.0 for src in sources}

    def update_from_loss(
        self,
        source_name: str,
        loss: float,
    ) -> None:
        """
        Update source weight based on observed loss.
        Higher loss → increase weight proportionally.
        """
        if source_name not in self._ema_losses:
            raise KeyError(f"Unknown source: {source_name!r}")

        alpha = self.adaptation_rate
        old_ema = self._ema_losses[source_name]
        new_ema = alpha * loss + (1 - alpha) * old_ema
        self._ema_losses[source_name] = new_ema

        # Update the mixer weight to reflect EMA loss (higher loss → higher weight)
        self.update_weights(source_name, new_ema)

    def get_adapted_weights(self) -> Dict[str, float]:
        """Return current adapted weights per source."""
        return {name: self._weights[name] for name in self._weights}
