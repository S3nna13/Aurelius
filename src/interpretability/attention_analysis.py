"""
src/interpretability/attention_analysis.py

Attention analysis utilities for extracting and quantifying patterns in
transformer attention weights for interpretability.

Pure PyTorch — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# AttentionConfig
# ---------------------------------------------------------------------------


@dataclass
class AttentionConfig:
    """Configuration for attention analysis experiments."""

    n_heads: int = 8
    n_layers: int = 12
    track_entropy: bool = True
    track_distance: bool = True


# ---------------------------------------------------------------------------
# compute_attention_entropy
# ---------------------------------------------------------------------------


def compute_attention_entropy(attn_weights: Tensor) -> Tensor:
    """
    Compute Shannon entropy per head per query position.

    Parameters
    ----------
    attn_weights : Tensor, shape (H, T, T)
        Attention weights (assumed to sum to 1 along last dim, i.e. softmaxed).

    Returns
    -------
    Tensor of shape (H, T) — entropy in nats for each (head, query) pair.
    """
    # H = -sum(p * log(p + eps)) over key dim
    # attn_weights: (H, T, T); sum over last dim (keys)
    eps = 1e-9
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # (H, T)
    return entropy


# ---------------------------------------------------------------------------
# compute_mean_attention_distance
# ---------------------------------------------------------------------------


def compute_mean_attention_distance(attn_weights: Tensor) -> Tensor:
    """
    Compute the weighted-average |i - j| distance per head per query.

    Parameters
    ----------
    attn_weights : Tensor, shape (H, T, T)

    Returns
    -------
    Tensor of shape (H, T) — mean distance for each (head, query) position.
    """
    H, T, _ = attn_weights.shape
    # Build position index tensor: positions[j] = j, shape (T,)
    positions = torch.arange(T, dtype=attn_weights.dtype, device=attn_weights.device)
    # For each query i, the key positions are 0..T-1; we want |i - j|
    # query_positions: (T, 1), key_positions: (1, T) → distances: (T, T)
    distances = (
        positions.unsqueeze(0) - positions.unsqueeze(1)
    ).abs()  # (T, T) where [i,j] = |i-j|
    # distances[i, j] = |i - j|; broadcast over heads: (H, T, T)
    distances = distances.unsqueeze(0).expand(H, T, T)  # (H, T, T)
    # Weighted average over key dim
    mean_dist = (attn_weights * distances).sum(dim=-1)  # (H, T)
    return mean_dist


# ---------------------------------------------------------------------------
# compute_head_agreement
# ---------------------------------------------------------------------------


def compute_head_agreement(attn1: Tensor, attn2: Tensor) -> Tensor:
    """
    Compute per-head cosine similarity between two sets of attention patterns.

    Parameters
    ----------
    attn1 : Tensor, shape (H, T, T)
    attn2 : Tensor, shape (H, T, T)

    Returns
    -------
    Tensor of shape (H,) — cosine similarity per head (in [-1, 1]).
    """
    H, T, _ = attn1.shape
    # Flatten T*T for each head
    flat1 = attn1.reshape(H, T * T)  # (H, T*T)
    flat2 = attn2.reshape(H, T * T)  # (H, T*T)

    dot = (flat1 * flat2).sum(dim=-1)  # (H,)
    norm1 = flat1.norm(dim=-1).clamp(min=1e-9)  # (H,)
    norm2 = flat2.norm(dim=-1).clamp(min=1e-9)  # (H,)
    return dot / (norm1 * norm2)  # (H,)


# ---------------------------------------------------------------------------
# find_diagonal_heads
# ---------------------------------------------------------------------------


def find_diagonal_heads(attn_weights: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Identify heads that predominantly attend to the current (self) token.

    A head is "diagonal" if the mean attention on the main diagonal
    exceeds *threshold*.

    Parameters
    ----------
    attn_weights : Tensor, shape (H, T, T)
    threshold    : float, default 0.5

    Returns
    -------
    Tensor of shape (H,), dtype bool.
    """
    H, T, _ = attn_weights.shape
    # Extract main diagonal for each head: attn_weights[:, i, i]
    # torch.diagonal returns shape (T, H) with offset=0 on last two dims
    diag = torch.diagonal(attn_weights, offset=0, dim1=1, dim2=2)  # (H, T)
    mean_diag = diag.mean(dim=-1)  # (H,)
    return mean_diag > threshold


# ---------------------------------------------------------------------------
# find_previous_token_heads
# ---------------------------------------------------------------------------


def find_previous_token_heads(attn_weights: Tensor, threshold: float = 0.3) -> Tensor:
    """
    Identify heads that predominantly attend to the immediately preceding token.

    For each head h, the "previous-token" score is the mean of
    attn_weights[h, i, i-1] over i in [1, T).

    Parameters
    ----------
    attn_weights : Tensor, shape (H, T, T)
    threshold    : float, default 0.3

    Returns
    -------
    Tensor of shape (H,), dtype bool.
    """
    H, T, _ = attn_weights.shape
    if T < 2:
        return torch.zeros(H, dtype=torch.bool, device=attn_weights.device)

    # attn_weights[:, i, i-1] for i in 1..T-1
    # Rows 1..T-1 (query), column i-1 (key) → super-diagonal of the key-offset matrix
    # Use indexing: query indices = [1, 2, ..., T-1], key indices = [0, 1, ..., T-2]
    query_idx = torch.arange(1, T, device=attn_weights.device)
    key_idx = torch.arange(0, T - 1, device=attn_weights.device)
    prev_attn = attn_weights[:, query_idx, key_idx]  # (H, T-1)
    mean_prev = prev_attn.mean(dim=-1)  # (H,)
    return mean_prev > threshold


# ---------------------------------------------------------------------------
# compute_attention_sink_score
# ---------------------------------------------------------------------------


def compute_attention_sink_score(attn_weights: Tensor, sink_position: int = 0) -> Tensor:
    """
    Compute the fraction of attention mass flowing to a "sink" position.

    For each head, this is the mean over queries of attn_weights[h, :, sink_position].

    Parameters
    ----------
    attn_weights  : Tensor, shape (H, T, T)
    sink_position : int, default 0  (typically the first token / BOS)

    Returns
    -------
    Tensor of shape (H,) — sink score per head, in [0, 1].
    """
    # attn_weights[:, :, sink_position] has shape (H, T)
    sink_attn = attn_weights[:, :, sink_position]  # (H, T)
    return sink_attn.mean(dim=-1)  # (H,)


# ---------------------------------------------------------------------------
# AttentionAnalyzer
# ---------------------------------------------------------------------------


class AttentionAnalyzer:
    """High-level API for attention pattern characterisation."""

    def __init__(self, config: AttentionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # analyze_layer
    # ------------------------------------------------------------------

    def analyze_layer(self, attn_weights: Tensor) -> dict[str, Tensor]:
        """
        Run all analyses on one layer's attention weights.

        Parameters
        ----------
        attn_weights : Tensor, shape (H, T, T)

        Returns
        -------
        dict with keys:
            "entropy"          : (H, T)
            "mean_distance"    : (H, T)
            "diagonal_heads"   : (H,) bool
            "prev_token_heads" : (H,) bool
            "sink_scores"      : (H,)
        """
        result: dict[str, Tensor] = {}

        result["entropy"] = compute_attention_entropy(attn_weights)
        result["mean_distance"] = compute_mean_attention_distance(attn_weights)
        result["diagonal_heads"] = find_diagonal_heads(attn_weights)
        result["prev_token_heads"] = find_previous_token_heads(attn_weights)
        result["sink_scores"] = compute_attention_sink_score(attn_weights)

        return result

    # ------------------------------------------------------------------
    # summarize
    # ------------------------------------------------------------------

    def summarize(self, analysis: dict[str, Tensor]) -> dict[str, float]:
        """
        Reduce an analysis dict to scalar summary statistics.

        Parameters
        ----------
        analysis : dict as returned by analyze_layer

        Returns
        -------
        dict with keys:
            "mean_entropy"    : float
            "mean_distance"   : float
            "n_diagonal"      : float (count of diagonal heads)
            "n_prev_token"    : float (count of previous-token heads)
            "max_sink_score"  : float
        """
        return {
            "mean_entropy": analysis["entropy"].mean().item(),
            "mean_distance": analysis["mean_distance"].mean().item(),
            "n_diagonal": analysis["diagonal_heads"].sum().item(),
            "n_prev_token": analysis["prev_token_heads"].sum().item(),
            "max_sink_score": analysis["sink_scores"].max().item(),
        }
