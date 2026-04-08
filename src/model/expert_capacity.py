"""Capacity accounting helpers for MoE experts."""

from __future__ import annotations

import torch


def expert_token_counts(assignments: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Count how many tokens are routed to each expert."""
    if assignments.dim() != 1:
        raise ValueError("assignments must be 1D")
    if torch.any(assignments < 0) or torch.any(assignments >= n_experts):
        raise ValueError("assignments contain out-of-range expert ids")
    counts = torch.zeros(n_experts, dtype=torch.long, device=assignments.device)
    return counts.scatter_add(0, assignments, torch.ones_like(assignments, dtype=torch.long))


def capacity_overflow(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Number of overflow tokens per expert above fixed capacity."""
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    return torch.clamp(counts - capacity, min=0)


def capacity_utilization(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Mean utilization fraction relative to capacity."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return (counts.float() / capacity).mean()

