"""Correlation helpers for comparing reward signals."""

from __future__ import annotations

import torch


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation between two 1D tensors."""
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be matching 1D tensors")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if denom.item() == 0.0:
        return torch.tensor(0.0, device=x.device)
    return (x_centered @ y_centered) / denom


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Assign average-free ordinal ranks to 1D values."""
    if values.dim() != 1:
        raise ValueError("values must be 1D")
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.float32, device=values.device)
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spearman rank correlation."""
    return pearson_correlation(rankdata(x), rankdata(y))


def reward_agreement(x: torch.Tensor, y: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Fraction of examples where reward signs agree around a threshold."""
    if x.shape != y.shape:
        raise ValueError("x and y must match")
    x_sign = x >= threshold
    y_sign = y >= threshold
    return (x_sign == y_sign).float().mean()
