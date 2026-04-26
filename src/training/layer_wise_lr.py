"""
Layer-Wise Learning Rate Decay (LLRD) for the Aurelius LLM project.

Assigns lower learning rates to deeper/earlier layers to prevent catastrophic
forgetting during fine-tuning. Layer 0 is the top (output-side) layer and
receives base_lr; layer n-1 is the bottom (input-side) layer and receives the
most decayed rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LLRDConfig:
    """Configuration for Layer-Wise Learning Rate Decay."""

    base_lr: float = 1e-3
    """Learning rate for the top (output-side) layer (layer index 0)."""

    decay_factor: float = 0.9
    """Multiplicative decay applied per layer going toward the input."""

    n_layers: int = 12
    """Number of transformer layers."""

    min_lr_ratio: float = 0.01
    """Floor as a fraction of base_lr (prevents lr from going to zero)."""

    embedding_lr_ratio: float = 0.1
    """Learning rate for embedding parameters as a fraction of base_lr."""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def compute_layer_lrs(config: LLRDConfig) -> list[float]:
    """Return per-layer learning rates from layer 0 (top) to layer n-1 (bottom).

    Layer 0 (closest to output) receives ``config.base_lr``.
    Layer i receives ``base_lr * decay_factor ** i``, clamped from below at
    ``base_lr * min_lr_ratio``.

    Args:
        config: LLRD configuration.

    Returns:
        List of length ``n_layers`` with learning rates in order [layer0, ..., layer_{n-1}].
    """
    min_lr = config.base_lr * config.min_lr_ratio
    lrs: list[float] = []
    for i in range(config.n_layers):
        lr = config.base_lr * (config.decay_factor**i)
        lr = max(lr, min_lr)
        lrs.append(lr)
    return lrs


def assign_layer_params(
    named_params: list[tuple[str, nn.Parameter]],
    layer_patterns: list[str],
) -> list[list[tuple[str, nn.Parameter]]]:
    """Group named parameters by layer using substring pattern matching.

    Args:
        named_params: List of (name, parameter) pairs from a model.
        layer_patterns: List of ``n_layers`` strings. A parameter is assigned
            to the first pattern whose string appears as a substring of the
            parameter name.  Parameters that match no pattern are placed in a
            final "other" group.

    Returns:
        List of ``n_layers + 1`` groups. Groups 0..n_layers-1 correspond to
        layers; group ``n_layers`` is the "other/unmatched" group.
    """
    n = len(layer_patterns)
    groups: list[list[tuple[str, nn.Parameter]]] = [[] for _ in range(n + 1)]

    for name, param in named_params:
        matched = False
        for layer_idx, pattern in enumerate(layer_patterns):
            if pattern in name:
                groups[layer_idx].append((name, param))
                matched = True
                break
        if not matched:
            groups[n].append((name, param))

    return groups


def build_llrd_param_groups(
    model: nn.Module,
    config: LLRDConfig,
    layer_patterns: list[str],
) -> list[dict]:
    """Build optimizer param groups with LLRD learning rates.

    - Groups 0..n_layers-1 use per-layer decayed learning rates.
    - The final "other" group (typically embeddings + head) uses
      ``base_lr * embedding_lr_ratio``.

    Args:
        model: The model whose parameters will be trained.
        config: LLRD configuration.
        layer_patterns: List of ``n_layers`` substring patterns, one per layer.

    Returns:
        List of dicts with keys ``"params"`` and ``"lr"``, ready for any
        ``torch.optim.Optimizer``.
    """
    layer_lrs = compute_layer_lrs(config)
    named_params = list(model.named_parameters())
    groups_by_layer = assign_layer_params(named_params, layer_patterns)

    param_groups: list[dict] = []

    for layer_idx, group in enumerate(groups_by_layer[:-1]):
        params = [p for _, p in group]
        param_groups.append({"params": params, "lr": layer_lrs[layer_idx]})

    # "other" group — embeddings + anything unmatched
    other_params = [p for _, p in groups_by_layer[-1]]
    embedding_lr = config.base_lr * config.embedding_lr_ratio
    param_groups.append({"params": other_params, "lr": embedding_lr})

    return param_groups


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class LayerWiseLRScheduler:
    """Applies cosine decay with linear warmup to each layer's learning rate.

    The scheduler scales each param group's lr according to its ratio relative
    to ``config.base_lr``, ensuring layer-wise differences are preserved
    throughout training.

    Args:
        optimizer: A ``torch.optim.Optimizer`` whose param groups already have
            an ``"lr"`` key set (e.g. built by :func:`build_llrd_param_groups`).
        config: LLRD configuration (provides ``base_lr`` as the reference).
        layer_lrs: Initial per-layer learning rates (length must equal the
            number of param groups in ``optimizer``).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: LLRDConfig,
        layer_lrs: list[float],
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.layer_lrs = layer_lrs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, global_step: int, warmup_steps: int, total_steps: int) -> None:
        """Update each param group's ``lr`` for the given training step.

        Schedule (per group):
        - Warmup phase (``global_step < warmup_steps``): linear ramp from 0
          to ``group_lr``.
        - Cosine decay phase: cosine annealing from ``group_lr`` toward 0.

        The group base lr (``group_lr``) is derived from ``layer_lrs`` scaled
        by ``config.base_lr`` so that ratios are preserved.

        Args:
            global_step: Current training step (0-indexed).
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
        """
        scale = self._compute_scale(global_step, warmup_steps, total_steps)
        for group_idx, group in enumerate(self.optimizer.param_groups):
            if group_idx < len(self.layer_lrs):
                group["lr"] = self.layer_lrs[group_idx] * scale
            else:
                group["lr"] = group["lr"] * scale

    def get_lrs(self) -> list[float]:
        """Return the current learning rate for each param group.

        Returns:
            List of floats, one per optimizer param group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scale(global_step: int, warmup_steps: int, total_steps: int) -> float:
        """Compute the lr scale factor for the current step."""
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and global_step < warmup_steps:
            return float(global_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Statistics utility
# ---------------------------------------------------------------------------


def compute_lr_ratio_stats(layer_lrs: list[float], base_lr: float) -> dict[str, float]:
    """Compute summary statistics of per-layer lr ratios relative to ``base_lr``.

    Args:
        layer_lrs: List of per-layer learning rates.
        base_lr: Reference learning rate (denominator for ratios).

    Returns:
        Dict with keys ``"min_ratio"``, ``"max_ratio"``, and ``"mean_ratio"``.
    """
    if not layer_lrs:
        return {"min_ratio": 0.0, "max_ratio": 0.0, "mean_ratio": 0.0}
    ratios = [lr / base_lr for lr in layer_lrs]
    return {
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "mean_ratio": sum(ratios) / len(ratios),
    }
