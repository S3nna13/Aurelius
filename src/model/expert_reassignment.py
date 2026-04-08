"""Helpers for reassigning overflowed tokens to backup experts."""

from __future__ import annotations

import torch


def backup_expert_indices(router_probs: torch.Tensor) -> torch.Tensor:
    """Return experts ranked from best to worst for each token."""
    if router_probs.dim() != 2:
        raise ValueError("router_probs must be 2D [tokens, experts]")
    return torch.argsort(router_probs, dim=-1, descending=True)


def reassign_overflowed_tokens(
    primary_assignments: torch.Tensor,
    backup_rankings: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """Reassign tokens to the first expert with remaining capacity.

    Tokens that cannot be reassigned are marked `-1`.
    """
    if primary_assignments.dim() != 1:
        raise ValueError("primary_assignments must be 1D")
    if backup_rankings.dim() != 2:
        raise ValueError("backup_rankings must be 2D")
    if capacities.dim() != 1:
        raise ValueError("capacities must be 1D")
    if backup_rankings.size(0) != primary_assignments.numel():
        raise ValueError("backup_rankings rows must match number of assignments")
    if backup_rankings.size(1) != capacities.numel():
        raise ValueError("backup_rankings cols must match number of capacities")

    usage = torch.zeros_like(capacities)
    reassigned = torch.full_like(primary_assignments, -1)
    for token_idx in range(primary_assignments.numel()):
        for expert in backup_rankings[token_idx].tolist():
            if usage[expert] < capacities[expert]:
                usage[expert] += 1
                reassigned[token_idx] = expert
                break
    return reassigned


def reassignment_success_rate(assignments: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens that ended up assigned to any expert."""
    if assignments.numel() == 0:
        return torch.tensor(0.0)
    return (assignments >= 0).float().mean()
