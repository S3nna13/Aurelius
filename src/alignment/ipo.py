"""Identity Preference Optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_shapes(*tensors: torch.Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) > 1:
        raise ValueError(f"All tensors must share the same shape, got {sorted(shapes)}")


@dataclass(frozen=True)
class IPOMetrics:
    loss: torch.Tensor
    preference_gap: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor


def ipo_rewards(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor | None = None,
    ref_rejected_logps: torch.Tensor | None = None,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IPO-style chosen/rejected rewards."""
    if ref_chosen_logps is None:
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
    if ref_rejected_logps is None:
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
    _validate_shapes(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    return chosen_reward, rejected_reward


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor | None = None,
    ref_rejected_logps: torch.Tensor | None = None,
    beta: float = 0.1,
    target_margin: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the IPO objective on preference pairs."""
    metrics = ipo_metrics(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        target_margin=target_margin,
        reduction=reduction,
    )
    return metrics.loss


def ipo_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor | None = None,
    ref_rejected_logps: torch.Tensor | None = None,
    beta: float = 0.1,
    target_margin: float | None = None,
    reduction: str = "mean",
) -> IPOMetrics:
    """Return IPO loss together with the underlying preference statistics."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if ref_chosen_logps is None:
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
    if ref_rejected_logps is None:
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
    _validate_shapes(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    chosen_reward, rejected_reward = ipo_rewards(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    preference_gap = (policy_chosen_logps - policy_rejected_logps) - (
        ref_chosen_logps - ref_rejected_logps
    )
    margin = target_margin if target_margin is not None else 1.0 / (2.0 * beta)
    losses = (preference_gap - margin) ** 2

    if reduction == "mean":
        loss = losses.mean()
    elif reduction == "sum":
        loss = losses.sum()
    elif reduction == "none":
        loss = losses
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return IPOMetrics(
        loss=loss,
        preference_gap=preference_gap,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
    )
