"""Context compression for long input sequences.

Reduces sequence length while preserving the most informative tokens,
enabling efficient inference over lengthy contexts without losing key
information.

Strategies implemented:
- Attention-score-based token selection (learnable query vector)
- L2-norm-based token scoring
- Strided subsampling
- Local average pooling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """Configuration for context compression.

    Attributes:
        target_ratio: Fraction of tokens to keep (0 < target_ratio <= 1).
        min_tokens: Minimum number of tokens to retain after compression.
        method: Scoring method to use — ``"attention_score"`` or ``"norm"``.
        stride: Step size used by :func:`strided_compression`.
    """
    target_ratio: float = 0.5
    min_tokens: int = 64
    method: str = "attention_score"
    stride: int = 2


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_tokens_by_attention(hidden: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Score each token by its dot-product relevance to a query vector.

    Args:
        hidden: Hidden states of shape ``(B, T, d)``.
        query:  Query vectors of shape ``(B, d)``.

    Returns:
        Scores of shape ``(B, T)``.
    """
    # query: (B, d) -> (B, d, 1) for batched matmul
    scores = torch.bmm(hidden, query.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return scores


def score_tokens_by_norm(hidden: torch.Tensor) -> torch.Tensor:
    """Score tokens by the L2 norm of their hidden state.

    Args:
        hidden: Hidden states of shape ``(B, T, d)``.

    Returns:
        Non-negative scores of shape ``(B, T)``.
    """
    return torch.linalg.norm(hidden, dim=-1)  # (B, T)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_top_tokens(
    scores: torch.Tensor,
    k: int,
    preserve_order: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-*k* tokens per sequence according to *scores*.

    Args:
        scores: Per-token scores of shape ``(B, T)``.
        k: Number of tokens to select.
        preserve_order: If ``True``, return selected indices in ascending
            (original) order.

    Returns:
        A tuple of:
        - ``indices`` — shape ``(B, k)``, the chosen token positions.
        - ``mask``    — shape ``(B, T)``, boolean mask with exactly *k*
          ``True`` entries per row.
    """
    B, T = scores.shape
    # topk_indices: (B, k) in descending score order
    _, topk_indices = torch.topk(scores, k, dim=-1)

    if preserve_order:
        # Sort each row so indices appear in ascending sequence order
        topk_indices, _ = torch.sort(topk_indices, dim=-1)

    # Build boolean mask
    mask = torch.zeros(B, T, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, topk_indices, True)

    return topk_indices, mask


# ---------------------------------------------------------------------------
# Structural compression
# ---------------------------------------------------------------------------

def strided_compression(hidden: torch.Tensor, stride: int) -> torch.Tensor:
    """Keep every *stride*-th token.

    Args:
        hidden: Hidden states of shape ``(B, T, d)``.
        stride: Subsampling step.

    Returns:
        Compressed states of shape ``(B, T // stride, d)``.
    """
    return hidden[:, ::stride, :]


def local_pooling_compression(hidden: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Average-pool tokens in non-overlapping windows of *pool_size*.

    Args:
        hidden: Hidden states of shape ``(B, T, d)``.
        pool_size: Window size for pooling.

    Returns:
        Pooled states of shape ``(B, T // pool_size, d)``.
    """
    B, T, d = hidden.shape
    n_windows = T // pool_size
    # Reshape to (B, n_windows, pool_size, d) then average over pool axis
    trimmed = hidden[:, : n_windows * pool_size, :]  # drop trailing tokens
    pooled = trimmed.reshape(B, n_windows, pool_size, d).mean(dim=2)
    return pooled


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class ContextCompressor(nn.Module):
    """Learnable context compressor using attention-score-based token selection.

    The module maintains a learnable query vector.  During :meth:`forward`
    it scores each token by its dot-product similarity to this query,
    selects the top-*k* tokens, and returns their hidden states together
    with a boolean selection mask.

    Args:
        d_model: Dimensionality of the hidden states.
        config:  :class:`CompressionConfig` instance controlling behaviour.
    """

    def __init__(self, d_model: int, config: CompressionConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config
        # Learnable query vector (d_model,)
        self.query = nn.Parameter(torch.randn(d_model))

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress *hidden* by selecting the most relevant tokens.

        Args:
            hidden: Input hidden states of shape ``(B, T, d_model)``.

        Returns:
            A tuple of:
            - ``compressed`` — shape ``(B, k, d_model)`` containing only the
              selected token representations.
            - ``mask`` — boolean tensor of shape ``(B, T)`` indicating which
              tokens were selected.

        Where ``k = max(config.min_tokens, int(T * config.target_ratio))``.
        """
        B, T, d = hidden.shape
        k = max(self.config.min_tokens, int(T * self.config.target_ratio))
        k = min(k, T)  # cannot select more tokens than available

        # Expand learnable query to batch: (B, d)
        query = self.query.unsqueeze(0).expand(B, -1)

        if self.config.method == "norm":
            scores = score_tokens_by_norm(hidden)
        else:  # default: "attention_score"
            scores = score_tokens_by_attention(hidden, query)

        indices, mask = select_top_tokens(scores, k, preserve_order=True)

        # Gather selected token hidden states: (B, k, d)
        idx_expanded = indices.unsqueeze(-1).expand(B, k, d)
        compressed = torch.gather(hidden, 1, idx_expanded)

        return compressed, mask


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_compression_ratio(original_len: int, compressed_len: int) -> float:
    """Return the ratio of compressed to original sequence length.

    Args:
        original_len:   Length of the uncompressed sequence.
        compressed_len: Length of the compressed sequence.

    Returns:
        ``compressed_len / original_len`` as a Python float.
    """
    return compressed_len / original_len


def reconstruct_from_mask(
    compressed: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Reconstruct the full-length sequence from compressed tokens and mask.

    Selected positions (``mask == True``) are filled with the corresponding
    rows of *compressed*; unselected positions receive *fill_value*.

    Args:
        compressed: Selected token representations, shape ``(B, k, d)``.
        mask:       Boolean selection mask, shape ``(B, T)``.
        fill_value: Scalar used to fill unselected positions.

    Returns:
        Reconstructed tensor of shape ``(B, T, d)``.
    """
    B, T = mask.shape
    _, k, d = compressed.shape

    output = torch.full((B, T, d), fill_value, dtype=compressed.dtype, device=compressed.device)
    # mask: (B, T) -> expand to (B, T, d) for scatter
    mask_expanded = mask.unsqueeze(-1).expand(B, T, d)  # (B, T, d)
    # compressed needs to be (B, T, d) shaped for masked_scatter;
    # masked_scatter fills in row-major order matching True positions
    output[mask_expanded] = compressed.reshape(-1)
    return output
