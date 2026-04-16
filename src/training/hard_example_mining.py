"""Hard Example Mining for LLM training.

Implements Online Hard Example Mining (OHEM), focal weighting, EMA-based
difficulty tracking, curriculum sampling, and related utilities to
upweight difficult training examples and focus learning on hard samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class MiningConfig:
    """Configuration for hard example mining."""

    top_k_ratio: float = 0.5
    """Fraction of samples with highest loss to keep in OHEM (0 < ratio <= 1.0)."""

    min_loss_threshold: float = 0.0
    """Minimum loss value; samples below this are never selected as hard."""

    use_focal: bool = False
    """If True, use focal weighting instead of OHEM selection."""

    focal_gamma: float = 2.0
    """Focal loss exponent gamma for down-weighting easy examples."""

    ema_decay: float = 0.99
    """Exponential moving average decay for per-sample loss tracking."""

    update_freq: int = 10
    """How often (in steps) the difficulty scores should be refreshed."""


# ---------------------------------------------------------------------------
# Core mining functions
# ---------------------------------------------------------------------------

def ohem_loss(per_sample_losses: Tensor, ratio: float = 0.5) -> Tensor:
    """Online Hard Example Mining: select top-ratio fraction with highest loss.

    Args:
        per_sample_losses: (N,) per-sample loss values.
        ratio: Fraction of samples to keep (by highest loss). Clamped to [0, 1].

    Returns:
        Scalar mean loss over the selected hard examples.
    """
    N = per_sample_losses.shape[0]
    ratio = float(max(0.0, min(1.0, ratio)))
    k = max(1, int(N * ratio))
    k = min(k, N)

    topk_vals, _ = torch.topk(per_sample_losses, k)
    return topk_vals.mean()


def focal_weight(per_sample_losses: Tensor, gamma: float = 2.0) -> Tensor:
    """Compute focal weights for per-sample losses.

    Focal weight = (1 - exp(-loss))^gamma, then normalized to sum to 1.
    Samples with high loss receive higher weight, suppressing easy examples.

    Args:
        per_sample_losses: (N,) per-sample loss values (non-negative).
        gamma: Exponent controlling how sharply easy samples are down-weighted.

    Returns:
        (N,) normalized focal weight tensor summing to 1.
    """
    # p_t proxy: exp(-loss) approximates model confidence for classification.
    # For regression-style losses we use the same formula.
    p = torch.exp(-per_sample_losses)           # (N,) in (0, 1]
    w = (1.0 - p) ** gamma                      # (N,) focal modulation
    total = w.sum()
    if total < 1e-12:
        # Degenerate case: uniform weights
        return torch.full_like(w, 1.0 / per_sample_losses.shape[0])
    return w / total


def compute_difficulty_scores(
    per_sample_losses: Tensor,
    ema_losses: Optional[Tensor] = None,
    ema_decay: float = 0.99,
) -> Tuple[Tensor, Tensor]:
    """Update EMA of per-sample losses and return current and updated EMA.

    If ``ema_losses`` is None, the EMA is initialized to the current losses.

    Args:
        per_sample_losses: (N,) current per-sample loss values.
        ema_losses: (N,) previous EMA loss estimates, or None.
        ema_decay: Decay factor alpha; new_ema = alpha * old + (1-alpha) * current.

    Returns:
        Tuple (current_losses, updated_ema), each shape (N,).
    """
    if ema_losses is None:
        updated_ema = per_sample_losses.clone().detach()
    else:
        updated_ema = ema_decay * ema_losses + (1.0 - ema_decay) * per_sample_losses.detach()
    return per_sample_losses, updated_ema


def rank_by_difficulty(
    losses: Tensor,
    percentile: float = 0.8,
) -> Dict[str, Tensor]:
    """Partition sample indices into hard, medium, and easy by loss percentile.

    Split points:
    - hard:   top (1 - percentile) fraction by loss (highest losses)
    - easy:   bottom percentile/2 fraction by loss (lowest losses)
    - medium: everything in between

    Args:
        losses: (N,) per-sample loss values.
        percentile: Threshold controlling the split (0 < percentile < 1).

    Returns:
        Dict with keys "hard", "medium", "easy", each containing a 1-D
        LongTensor of indices (sorted by descending loss within each group).
    """
    N = losses.shape[0]
    sorted_indices = torch.argsort(losses, descending=True)   # highest loss first

    n_hard = max(1, int(N * (1.0 - percentile)))
    n_easy = max(1, int(N * (percentile / 2.0)))
    # Ensure groups don't exceed N
    n_hard = min(n_hard, N)
    n_easy = min(n_easy, N - n_hard)
    n_medium = N - n_hard - n_easy

    hard_indices = sorted_indices[:n_hard]
    medium_indices = sorted_indices[n_hard: n_hard + n_medium]
    easy_indices = sorted_indices[n_hard + n_medium:]

    return {
        "hard": hard_indices,
        "medium": medium_indices,
        "easy": easy_indices,
    }


# ---------------------------------------------------------------------------
# HardExampleSampler
# ---------------------------------------------------------------------------

class HardExampleSampler:
    """Maintains per-sample EMA losses and provides sampling strategies.

    Args:
        config: MiningConfig with EMA decay and other parameters.
        dataset_size: Total number of samples in the dataset.
    """

    def __init__(self, config: MiningConfig, dataset_size: int) -> None:
        self.config = config
        self.dataset_size = dataset_size
        # Initialize EMA losses to zero; will be updated incrementally.
        self.ema_losses: Tensor = torch.zeros(dataset_size)

    def update(self, indices: Tensor, losses: Tensor) -> None:
        """Update EMA loss estimates for specific sample indices.

        Args:
            indices: (M,) LongTensor of sample indices to update.
            losses: (M,) corresponding current loss values.
        """
        alpha = self.config.ema_decay
        self.ema_losses[indices] = (
            alpha * self.ema_losses[indices] + (1.0 - alpha) * losses.detach().to(self.ema_losses.device)
        )

    def sample_hard_indices(self, n: int) -> Tensor:
        """Return the n sample indices with the highest EMA loss.

        Args:
            n: Number of indices to return.

        Returns:
            (n,) LongTensor of sample indices, sorted by descending EMA loss.
        """
        n = min(n, self.dataset_size)
        _, top_indices = torch.topk(self.ema_losses, n)
        return top_indices

    def sample_curriculum(self, n: int, epoch: int, total_epochs: int) -> Tensor:
        """Sample n indices following a curriculum schedule.

        Early epochs favour easy examples (low EMA loss); late epochs favour
        hard examples (high EMA loss). Transition happens around the midpoint.

        The interpolation uses a sigmoid-like ramp:
        - epoch < midpoint: sample from the easy half
        - epoch >= midpoint: sample from the hard half

        Args:
            n: Number of indices to return.
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of training epochs.

        Returns:
            (n,) LongTensor of selected sample indices.
        """
        n = min(n, self.dataset_size)
        midpoint = max(1, total_epochs // 2)

        sorted_indices = torch.argsort(self.ema_losses, descending=False)  # low → high loss

        if epoch < midpoint:
            # Early: prefer easy (low-loss) examples
            pool = sorted_indices[: max(n, self.dataset_size // 2)]
        else:
            # Late: prefer hard (high-loss) examples — reverse to descending
            pool = sorted_indices.flip(0)[: max(n, self.dataset_size // 2)]

        # Sample n from pool (without replacement if possible)
        if pool.shape[0] <= n:
            return pool
        perm = torch.randperm(pool.shape[0])[:n]
        return pool[perm]


# ---------------------------------------------------------------------------
# Unified loss with mining
# ---------------------------------------------------------------------------

def loss_with_mining(per_sample_losses: Tensor, config: MiningConfig) -> Tensor:
    """Compute a scalar loss using the mining strategy specified in config.

    If ``config.use_focal`` is True, returns the focal-weighted sum of losses.
    Otherwise, returns the OHEM mean over the top-k fraction.

    Args:
        per_sample_losses: (N,) per-sample loss values.
        config: MiningConfig controlling which strategy to apply.

    Returns:
        Scalar loss tensor.
    """
    if config.use_focal:
        weights = focal_weight(per_sample_losses, gamma=config.focal_gamma)
        return (weights * per_sample_losses).sum()
    else:
        return ohem_loss(per_sample_losses, ratio=config.top_k_ratio)
