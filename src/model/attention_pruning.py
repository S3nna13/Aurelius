"""Attention head importance scoring and pruning for efficient transformer inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PruningConfig:
    """Configuration for attention head pruning."""

    n_heads: int = 8
    """Number of attention heads in the model."""

    d_head: int = 64
    """Dimension per attention head."""

    prune_ratio: float = 0.5
    """Fraction of heads to remove (0.5 = remove bottom 50% by importance)."""

    importance_metric: str = "magnitude"
    """Importance scoring method: 'magnitude' | 'gradient' | 'random'."""


# ---------------------------------------------------------------------------
# Per-head importance computation
# ---------------------------------------------------------------------------

def compute_head_importance_magnitude(
    weight: Tensor,
    n_heads: int,
    d_head: int,
) -> Tensor:
    """Compute per-head L2 norm importance from a Q or K projection weight.

    Args:
        weight: Shape (n_heads * d_head, d_model) — the projection weight matrix.
        n_heads: Number of attention heads.
        d_head: Dimension per head.

    Returns:
        Importance scores of shape (n_heads,) — L2 norm for each head's rows.
    """
    # Reshape to (n_heads, d_head, d_model) then take Frobenius norm per head
    d_model = weight.shape[1]
    w = weight.view(n_heads, d_head, d_model)   # (n_heads, d_head, d_model)
    return w.norm(dim=(1, 2))                    # (n_heads,)


def compute_head_importance_random(n_heads: int, seed: int = 42) -> Tensor:
    """Return random per-head importance scores for ablation baselines.

    Args:
        n_heads: Number of attention heads.
        seed: Manual seed for reproducibility.

    Returns:
        Random importance scores of shape (n_heads,).
    """
    torch.manual_seed(seed)
    return torch.rand(n_heads)


# ---------------------------------------------------------------------------
# Pruning index / mask helpers
# ---------------------------------------------------------------------------

def get_heads_to_prune(importance: Tensor, prune_ratio: float) -> List[int]:
    """Return the indices of heads to prune (least important first).

    Args:
        importance: Shape (n_heads,) — importance score per head.
        prune_ratio: Fraction of heads to remove (e.g. 0.5 removes 50%).

    Returns:
        Sorted list of head indices to prune (ascending order).
    """
    n_heads = importance.shape[0]
    n_prune = math.floor(prune_ratio * n_heads)
    if n_prune == 0:
        return []
    # argsort ascending — take the bottom n_prune
    sorted_indices = importance.argsort()          # ascending by importance
    pruned = sorted_indices[:n_prune].tolist()
    return sorted(int(i) for i in pruned)


def create_head_mask(n_heads: int, heads_to_prune: List[int]) -> Tensor:
    """Create a boolean keep-mask for attention heads.

    Args:
        n_heads: Total number of heads.
        heads_to_prune: List of head indices to prune.

    Returns:
        Bool tensor of shape (n_heads,): True = keep, False = prune.
    """
    mask = torch.ones(n_heads, dtype=torch.bool)
    for idx in heads_to_prune:
        mask[idx] = False
    return mask


def apply_head_mask(attn_output: Tensor, head_mask: Tensor) -> Tensor:
    """Zero out pruned heads in a multi-head attention output tensor.

    Args:
        attn_output: Shape (B, n_heads, T, d_head).
        head_mask: Shape (n_heads,) bool tensor — True = keep, False = prune.

    Returns:
        Tensor of the same shape as attn_output with pruned heads zeroed.
    """
    # Broadcast mask: (1, n_heads, 1, 1)
    mask = head_mask.to(dtype=attn_output.dtype, device=attn_output.device)
    mask = mask.view(1, -1, 1, 1)
    return attn_output * mask


# ---------------------------------------------------------------------------
# HeadPruner class
# ---------------------------------------------------------------------------

class HeadPruner:
    """High-level helper that encapsulates importance analysis and mask creation."""

    def __init__(self, config: PruningConfig) -> None:
        self.config = config

    def analyze_heads(self, weight: Tensor) -> Tensor:
        """Compute per-head importance scores using the configured metric.

        Args:
            weight: Q or K projection weight of shape (n_heads * d_head, d_model).

        Returns:
            Importance scores of shape (n_heads,).
        """
        metric = self.config.importance_metric
        if metric == "magnitude":
            return compute_head_importance_magnitude(
                weight, self.config.n_heads, self.config.d_head
            )
        elif metric == "random":
            return compute_head_importance_random(self.config.n_heads)
        else:
            raise ValueError(
                f"Unknown importance_metric '{metric}'. "
                "Choose from 'magnitude', 'gradient', 'random'."
            )

    def get_pruning_mask(self, weight: Tensor) -> Tensor:
        """Compute importance and return the boolean keep-mask.

        Args:
            weight: Q or K projection weight of shape (n_heads * d_head, d_model).

        Returns:
            Bool tensor of shape (n_heads,): True = keep, False = prune.
        """
        importance = self.analyze_heads(weight)
        heads_to_prune = get_heads_to_prune(importance, self.config.prune_ratio)
        return create_head_mask(self.config.n_heads, heads_to_prune)

    def prune_step(self, attn_output: Tensor, weight: Tensor) -> Tensor:
        """Analyze heads and apply the resulting mask in a single call.

        Args:
            attn_output: Shape (B, n_heads, T, d_head).
            weight: Q or K projection weight of shape (n_heads * d_head, d_model).

        Returns:
            Masked attn_output with pruned heads zeroed, same shape as input.
        """
        mask = self.get_pruning_mask(weight)
        return apply_head_mask(attn_output, mask)


# ---------------------------------------------------------------------------
# Sparsity utility
# ---------------------------------------------------------------------------

def compute_sparsity_ratio(mask: Tensor) -> float:
    """Return the fraction of heads that are pruned (masked out).

    Args:
        mask: Bool tensor of shape (n_heads,): True = keep, False = prune.

    Returns:
        Float in [0, 1] — fraction of heads that are False (pruned).
    """
    n_heads = mask.numel()
    n_pruned = (~mask).sum().item()
    return n_pruned / n_heads if n_heads > 0 else 0.0


# ---------------------------------------------------------------------------
# PrunedMultiHeadAttention nn.Module
# ---------------------------------------------------------------------------

class PrunedMultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional static head pruning via a mask.

    Pruned heads are zeroed before the output projection, keeping model shape
    fixed so checkpoints stay compatible after pruning.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads. d_model must be divisible by n_heads.
        head_mask: Optional bool tensor of shape (n_heads,). True = keep.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Register mask as a buffer so it moves with .to(device)
        if head_mask is None:
            head_mask = torch.ones(n_heads, dtype=torch.bool)
        self.register_buffer("head_mask", head_mask)

    def set_head_mask(self, mask: Tensor) -> None:
        """Update the active head mask.

        Args:
            mask: Bool tensor of shape (n_heads,). True = keep, False = prune.
        """
        assert mask.shape == (self.n_heads,), (
            f"Expected mask of shape ({self.n_heads},), got {mask.shape}"
        )
        self.head_mask = mask.to(dtype=torch.bool, device=self.head_mask.device)

    def forward(self, x: Tensor) -> Tensor:
        """Compute masked multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        scale = math.sqrt(Dh)

        # Projections
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)                    # (B, H, T, Dh)

        # Apply head mask — zero pruned heads
        attn_out = apply_head_mask(attn_out, self.head_mask)

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_out)
