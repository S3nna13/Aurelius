"""Scoring helpers for draft-tree paths."""

from __future__ import annotations

import torch


def cumulative_path_score(node_scores: torch.Tensor) -> torch.Tensor:
    """Sum scores along a single draft path."""
    if node_scores.dim() != 1:
        raise ValueError("node_scores must be 1D")
    return node_scores.sum()


def normalized_path_score(node_scores: torch.Tensor) -> torch.Tensor:
    """Average score along a draft path."""
    if node_scores.dim() != 1:
        raise ValueError("node_scores must be 1D")
    if node_scores.numel() == 0:
        return torch.tensor(0.0)
    return node_scores.mean()


def best_path_index(path_scores: torch.Tensor) -> int:
    """Index of the highest-scoring path."""
    if path_scores.dim() != 1:
        raise ValueError("path_scores must be 1D")
    if path_scores.numel() == 0:
        raise ValueError("path_scores must be non-empty")
    return int(torch.argmax(path_scores).item())
