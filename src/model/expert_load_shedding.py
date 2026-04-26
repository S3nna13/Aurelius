"""Helpers for shedding overflow tokens from overloaded experts."""

from __future__ import annotations

import torch


def expert_overflow(assignments: torch.Tensor, n_experts: int, capacity: int) -> torch.Tensor:
    """Return per-expert overflow counts above capacity."""
    if assignments.dim() != 1:
        raise ValueError("assignments must be 1D")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if torch.any(assignments < 0) or torch.any(assignments >= n_experts):
        raise ValueError("assignments contain out-of-range expert ids")
    counts = torch.bincount(assignments, minlength=n_experts)
    return torch.clamp(counts - capacity, min=0)


def shed_overflow_tokens(assignments: torch.Tensor, n_experts: int, capacity: int) -> torch.Tensor:
    """Mark overflowed tokens with `-1` after each expert reaches capacity."""
    overflow = torch.zeros_like(assignments)
    kept = torch.zeros(n_experts, dtype=torch.long, device=assignments.device)
    output = assignments.clone()
    for idx, expert in enumerate(assignments.tolist()):
        if kept[expert] >= capacity:
            output[idx] = -1
            overflow[idx] = 1
        else:
            kept[expert] += 1
    return output


def kept_fraction(assignments: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens kept after load shedding."""
    if assignments.numel() == 0:
        return torch.tensor(0.0)
    return (assignments >= 0).float().mean()
