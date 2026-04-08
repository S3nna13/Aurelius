"""Token importance scoring and dynamic pruning for efficient inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TokenPruningConfig:
    """Configuration for token pruning."""

    keep_ratio: float = 0.5
    """Fraction of tokens to keep (0.5 = keep 50%)."""

    scoring_method: str = "attention"
    """Scoring method: 'attention' | 'gradient' | 'random'."""

    min_tokens: int = 1
    """Always keep at least this many tokens."""

    protect_positions: list[int] = field(default_factory=list)
    """Always keep these token indices regardless of score."""


def score_tokens_by_attention(attention_weights: Tensor) -> Tensor:
    """Score tokens by how much other tokens attend to them.

    Args:
        attention_weights: Shape (B, n_heads, T, T). attention_weights[b, h, i, j]
            is how much token i attends to token j.

    Returns:
        Importance scores of shape (B, T) — mean over heads of column sums.
    """
    # Column sum along the query dimension (dim=2) gives total attention received
    # attention_weights: (B, n_heads, T_query, T_key)
    col_sums = attention_weights.sum(dim=2)  # (B, n_heads, T)
    return col_sums.mean(dim=1)              # (B, T)


def score_tokens_by_gradient(hidden_states: Tensor, grad: Tensor) -> Tensor:
    """Score tokens via gradient × activation magnitude.

    Importance = ||hidden * grad||_2 per token position.

    Args:
        hidden_states: Shape (B, T, D).
        grad: Shape (B, T, D) — gradient of loss w.r.t. hidden_states.

    Returns:
        Importance scores of shape (B, T).
    """
    product = hidden_states * grad          # (B, T, D)
    return product.norm(dim=-1)             # (B, T)


def select_important_tokens(scores: Tensor, config: TokenPruningConfig) -> Tensor:
    """Select the most important tokens based on scores.

    Args:
        scores: Shape (B, T) — importance score per position.
        config: Pruning configuration.

    Returns:
        Boolean mask of shape (B, T), True = keep.
    """
    B, T = scores.shape
    k = max(config.min_tokens, math.ceil(config.keep_ratio * T))
    k = min(k, T)  # can't keep more than T tokens

    # Start with all-False mask
    mask = torch.zeros(B, T, dtype=torch.bool, device=scores.device)

    # Always protect specified positions
    for pos in config.protect_positions:
        if 0 <= pos < T:
            mask[:, pos] = True

    # For each batch item, select top-k by score
    topk_indices = scores.topk(k, dim=1).indices  # (B, k)
    mask.scatter_(1, topk_indices, True)

    return mask


class TokenPruningLayer(nn.Module):
    """Prune tokens based on attention patterns, zeroing out pruned positions."""

    def __init__(self, config: TokenPruningConfig) -> None:
        super().__init__()
        self.config = config
        self._last_scores: Tensor | None = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Prune tokens, zeroing out non-kept positions.

        Args:
            hidden_states: Shape (B, T, D).
            attention_weights: Optional shape (B, n_heads, T, T).

        Returns:
            (pruned_hidden, keep_mask) — pruned_hidden same shape as input,
            keep_mask shape (B, T).
        """
        B, T, D = hidden_states.shape

        if attention_weights is not None:
            scores = score_tokens_by_attention(attention_weights)
        else:
            # Uniform scores — all tokens equally important
            scores = torch.ones(B, T, device=hidden_states.device,
                                dtype=hidden_states.dtype)

        self._last_scores = scores
        keep_mask = select_important_tokens(scores, self.config)

        # Zero out pruned positions (keep shape consistent)
        pruned_hidden = hidden_states * keep_mask.unsqueeze(-1).to(hidden_states.dtype)
        return pruned_hidden, keep_mask


class AdaptiveTokenPruner(nn.Module):
    """Learns to predict which tokens to prune via a small linear scorer."""

    def __init__(self, d_model: int, config: TokenPruningConfig) -> None:
        super().__init__()
        self.config = config
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Score, select, and zero out unimportant tokens.

        Args:
            hidden_states: Shape (B, T, D).

        Returns:
            (pruned_hidden, keep_mask, scores) — all with consistent batch dims.
            pruned_hidden: (B, T, D), keep_mask: (B, T), scores: (B, T).
        """
        scores = self.scorer(hidden_states).squeeze(-1)  # (B, T)
        keep_mask = select_important_tokens(scores, self.config)

        pruned_hidden = hidden_states * keep_mask.unsqueeze(-1).to(hidden_states.dtype)
        return pruned_hidden, keep_mask, scores

    def pruning_loss(self, scores: Tensor, keep_mask: Tensor) -> Tensor:
        """Sparsity regularization: push pruned-token scores toward zero.

        Args:
            scores: Shape (B, T) — learned importance scores.
            keep_mask: Shape (B, T) — True = kept.

        Returns:
            Scalar regularization loss.
        """
        pruned_scores = scores[~keep_mask]
        if pruned_scores.numel() == 0:
            return scores.new_tensor(0.0)
        return pruned_scores.mean()


def evaluate_pruning_efficiency(
    original_seq_len: int,
    keep_ratio: float,
) -> dict[str, float]:
    """Estimate efficiency gains from token pruning.

    Theoretical speedup assumes attention complexity is O(T^2), so reducing
    sequence length by keep_ratio gives speedup ≈ 1 / keep_ratio^2.

    Args:
        original_seq_len: Original number of tokens T.
        keep_ratio: Fraction of tokens to keep.

    Returns:
        Dict with "kept_tokens", "pruned_tokens", "theoretical_speedup".
    """
    kept = math.ceil(keep_ratio * original_seq_len)
    pruned = original_seq_len - kept
    speedup = 1.0 / (keep_ratio ** 2) if keep_ratio > 0 else float("inf")
    return {
        "kept_tokens": float(kept),
        "pruned_tokens": float(pruned),
        "theoretical_speedup": speedup,
    }
