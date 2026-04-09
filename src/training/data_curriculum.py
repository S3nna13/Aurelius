"""Multi-source data curriculum: loss-adaptive source weights with diversity regularization."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataSource:
    name: str
    weight: float = 1.0       # initial sampling weight
    domain: str = "general"   # for diversity grouping
    difficulty: float = 0.5   # [0, 1] for difficulty-based curriculum


@dataclass
class CurriculumConfig:
    n_sources: int                          # number of data sources
    update_interval: int = 100              # steps between weight updates
    loss_ema_alpha: float = 0.1             # EMA smoothing for per-source loss
    diversity_weight: float = 0.1           # regularization toward uniform domain sampling
    difficulty_warmup_steps: int = 1000     # ramp from easy to hard over this many steps
    min_source_weight: float = 0.05         # floor to prevent source starvation


def normalize_weights(weights: torch.Tensor, min_weight: float = 0.05) -> torch.Tensor:
    """Clip each weight to [min_weight, inf], then normalize to sum to 1.

    Args:
        weights: 1-D tensor of raw weights.
        min_weight: floor value applied before normalization.

    Returns:
        Normalized weight tensor of the same shape.
    """
    clipped = weights.clamp(min=min_weight)
    total = clipped.sum()
    if total <= 0:
        return torch.full_like(clipped, 1.0 / clipped.numel())
    return clipped / total


def compute_loss_adaptive_weights(
    source_losses: torch.Tensor,
    current_weights: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Update source weights so that higher-loss sources receive more attention.

    New weight is EMA-blended: new_w = alpha * losses + (1 - alpha) * current_weights,
    then normalized.

    Args:
        source_losses: 1-D tensor of per-source loss values.
        current_weights: 1-D tensor of current source weights (same length).
        alpha: EMA blending coefficient (0 = keep old weights, 1 = use raw losses).

    Returns:
        Normalized weight tensor of the same shape.
    """
    new_w = alpha * source_losses + (1.0 - alpha) * current_weights
    return normalize_weights(new_w)


def compute_difficulty_mask(
    sources: list[DataSource],
    step: int,
    warmup_steps: int,
) -> torch.Tensor:
    """Compute a binary inclusion mask based on curriculum difficulty progress.

    Progress ramps from 0 to 1 over warmup_steps.  A source is included when
    its difficulty <= progress + 0.1 (so very easy sources are always included).

    Args:
        sources: list of DataSource objects.
        step: current training step.
        warmup_steps: total steps over which difficulty ramps up.

    Returns:
        Binary mask tensor of shape (n_sources,) — 1.0 included, 0.0 excluded.
    """
    progress = min(step / max(warmup_steps, 1), 1.0)
    threshold = progress + 0.1
    mask = torch.tensor(
        [1.0 if src.difficulty <= threshold else 0.0 for src in sources],
        dtype=torch.float32,
    )
    return mask


def diversity_regularization(
    weights: torch.Tensor,
    sources: list[DataSource],
    diversity_weight: float,
) -> torch.Tensor:
    """Push source weights toward uniform distribution over domains.

    Target weight for each source = 1 / n_domains (equal share per domain,
    distributed uniformly among sources within that domain).

    Regularized weight = (1 - diversity_weight) * weights + diversity_weight * target_weights,
    then normalized.

    Args:
        weights: 1-D tensor of current source weights.
        sources: list of DataSource objects (same order as weights).
        diversity_weight: strength of regularization in [0, 1].

    Returns:
        Normalized regularized weight tensor.
    """
    domains = list({src.domain for src in sources})
    n_domains = len(domains)

    # Compute per-domain target weight = 1 / n_domains, split evenly among domain members
    domain_members: dict[str, list[int]] = {d: [] for d in domains}
    for i, src in enumerate(sources):
        domain_members[src.domain].append(i)

    target = torch.zeros(len(sources))
    for d, indices in domain_members.items():
        share = 1.0 / (n_domains * len(indices))
        for i in indices:
            target[i] = share

    regularized = (1.0 - diversity_weight) * weights + diversity_weight * target
    return normalize_weights(regularized)


class AdaptiveCurriculumSampler:
    """Adaptive multi-source curriculum sampler with loss-driven and diversity-aware weights.

    Maintains per-source EMA losses and updates sampling weights every
    config.update_interval steps.  Difficulty masking gradually unlocks harder
    sources over the warmup period.
    """

    def __init__(self, sources: list[DataSource], config: CurriculumConfig) -> None:
        self.sources = sources
        self.config = config

        # Initialize weights from DataSource.weight, then normalize
        raw = torch.tensor([s.weight for s in sources], dtype=torch.float32)
        self.weights: torch.Tensor = normalize_weights(raw, config.min_source_weight)

        # EMA loss estimates — start at 1.0 (neutral)
        self._ema_losses: dict[str, float] = {s.name: 1.0 for s in sources}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_weights(self, source_losses: dict[str, float], step: int) -> None:
        """Update per-source EMA losses and recompute sampling weights.

        Steps:
        1. Update EMA losses for each source present in source_losses.
        2. Compute loss-adaptive weights.
        3. Apply difficulty mask (zero out sources not yet unlocked).
        4. Apply diversity regularization.
        5. Store result in self.weights.

        Args:
            source_losses: mapping {source_name: loss_value} for the current step.
            step: current training step (used for difficulty curriculum).
        """
        alpha = self.config.loss_ema_alpha

        # 1. Update EMA losses
        for name, loss in source_losses.items():
            if name in self._ema_losses:
                old = self._ema_losses[name]
                self._ema_losses[name] = alpha * loss + (1.0 - alpha) * old

        # 2. Build loss tensor aligned with self.sources order
        loss_tensor = torch.tensor(
            [self._ema_losses[s.name] for s in self.sources], dtype=torch.float32
        )

        # 3. Loss-adaptive weights
        new_weights = compute_loss_adaptive_weights(loss_tensor, self.weights, alpha)

        # 4. Difficulty mask
        mask = compute_difficulty_mask(self.sources, step, self.config.difficulty_warmup_steps)
        masked = new_weights * mask

        # If all masked out (shouldn't happen normally), fall back to uniform of unmasked
        if masked.sum() <= 0:
            masked = mask.clone()
        if masked.sum() <= 0:
            masked = torch.ones(len(self.sources))

        # 5. Diversity regularization
        regularized = diversity_regularization(
            masked, self.sources, self.config.diversity_weight
        )

        # Apply min weight floor only to included (non-zero mask) sources then re-normalize
        # Sources with mask=0 stay at 0 after diversity_regularization may still assign them weight;
        # re-apply mask to enforce hard exclusion, then normalize.
        final = regularized * (mask > 0).float()
        if final.sum() <= 0:
            final = torch.ones(len(self.sources))

        self.weights = normalize_weights(final, self.config.min_source_weight * (mask > 0).float().max().item())

    def sample_source(self) -> DataSource:
        """Sample one DataSource proportionally to current weights.

        Returns:
            The sampled DataSource.
        """
        idx = torch.multinomial(self.weights, num_samples=1).item()
        return self.sources[int(idx)]

    def get_weights(self) -> dict[str, float]:
        """Return current weights keyed by source name.

        Returns:
            {source.name: weight} for all sources.
        """
        return {src.name: self.weights[i].item() for i, src in enumerate(self.sources)}
