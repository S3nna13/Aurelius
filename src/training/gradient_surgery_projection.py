"""Gradient Surgery projection for multi-task learning.

Implements PCGrad from "Gradient Surgery for Multi-Task Learning"
(arXiv:2001.06782) using the paper's gradient notation:

    g_i     : task-i gradient
    g_i_pc  : task-i gradient after projection of conflicts
    g       : aggregated gradient
    pi      : random task order used for pairwise projections
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GradientSurgeryProjectionConfig:
    """Configuration for PCGrad projection."""

    reduction: str = "mean"
    stability_eps: float = 1e-12
    implementation: str = "optimized"


@dataclass
class GradientSurgeryProjectionOutput:
    """Projected multi-task gradient state."""

    loss: Tensor
    g: Tensor
    g_i: Tensor
    g_i_pc: Tensor
    pi: Tensor


def _as_parameter_list(parameters: Iterable[Tensor]) -> list[Tensor]:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("parameters must contain at least one trainable tensor")
    return params


def _validate_task_losses(task_losses: Sequence[Tensor]) -> None:
    if len(task_losses) == 0:
        raise ValueError("task_losses must be non-empty")
    for index, loss in enumerate(task_losses):
        if loss.ndim != 0:
            raise ValueError(f"task_losses[{index}] must be a scalar tensor")


def _flatten_grads(gradients: Sequence[Tensor]) -> Tensor:
    return torch.cat([gradient.reshape(-1) for gradient in gradients], dim=0)


def collect_task_gradients(
    task_losses: Sequence[Tensor],
    parameters: Iterable[Tensor],
) -> Tensor:
    """Collect flattened per-task gradients into `g_i` with shape `(T, P)`."""

    _validate_task_losses(task_losses)
    params = _as_parameter_list(parameters)
    g_i: list[Tensor] = []

    for index, loss in enumerate(task_losses):
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=index < len(task_losses) - 1,
            create_graph=False,
            allow_unused=False,
        )
        g_i.append(_flatten_grads(gradients))

    return torch.stack(g_i, dim=0)


def sample_projection_order(T: int, device: torch.device) -> Tensor:
    """Sample the random task order `pi` used by PCGrad."""

    if T <= 0:
        raise ValueError("T must be positive")
    return torch.stack([torch.randperm(T, device=device) for _ in range(T)], dim=0)


def pcgrad_project_reference(g_i: Tensor, pi: Tensor, stability_eps: float = 1e-12) -> Tensor:
    """Reference PCGrad implementation from the paper's pairwise update rule."""

    if g_i.ndim != 2:
        raise ValueError("g_i must have shape (T, P)")
    if pi.shape != (g_i.size(0), g_i.size(0)):
        raise ValueError("pi must have shape (T, T)")

    T = g_i.size(0)
    g_i_pc = g_i.clone()
    for i in range(T):
        for j in pi[i].tolist():
            if i == j:
                continue
            dot = torch.dot(g_i_pc[i], g_i[j])
            if dot < 0:
                denom = torch.dot(g_i[j], g_i[j]) + stability_eps
                g_i_pc[i] = g_i_pc[i] - (dot / denom) * g_i[j]
    return g_i_pc


def pcgrad_project(g_i: Tensor, pi: Tensor, stability_eps: float = 1e-12) -> Tensor:
    """PCGrad projection using stacked tensor updates."""

    if g_i.ndim != 2:
        raise ValueError("g_i must have shape (T, P)")
    if pi.shape != (g_i.size(0), g_i.size(0)):
        raise ValueError("pi must have shape (T, T)")

    T = g_i.size(0)
    g_i_pc = g_i.clone()
    norm_sq = (g_i * g_i).sum(dim=1)

    for i in range(T):
        for j in pi[i]:
            j_index = int(j.item())
            if i == j_index:
                continue
            dot = (g_i_pc[i] * g_i[j_index]).sum()
            if dot < 0:
                g_i_pc[i] = g_i_pc[i] - (dot / (norm_sq[j_index] + stability_eps)) * g_i[j_index]
    return g_i_pc


def aggregate_projected_gradients(g_i_pc: Tensor, reduction: str = "mean") -> Tensor:
    """Aggregate projected task gradients into `g`."""

    if reduction == "mean":
        return g_i_pc.mean(dim=0)
    if reduction == "sum":
        return g_i_pc.sum(dim=0)
    raise ValueError("reduction must be 'mean' or 'sum'")


def build_gradient_surrogate_loss(g: Tensor, parameters: Iterable[Tensor]) -> Tensor:
    """Build a scalar surrogate whose backward pass writes `g` into `.grad`."""

    params = _as_parameter_list(parameters)
    pieces: list[Tensor] = []
    offset = 0
    for parameter in params:
        numel = parameter.numel()
        g_parameter = g[offset : offset + numel].view_as(parameter).detach()
        pieces.append((parameter * g_parameter).sum())
        offset += numel

    if offset != g.numel():
        raise ValueError("g has incompatible size for the provided parameters")

    return torch.stack(pieces).sum()


def gradient_surgery_projection(
    task_losses: Sequence[Tensor],
    parameters: Iterable[Tensor],
    config: GradientSurgeryProjectionConfig | None = None,
) -> GradientSurgeryProjectionOutput:
    """Project conflicting task gradients and return a backward-ready surrogate.

    The paper's update is:

        g_i_pc = g_i - proj_{g_j}(g_i)    when <g_i, g_j> < 0
        g = (1 / T) * sum_i g_i_pc
    """

    cfg = config or GradientSurgeryProjectionConfig()
    params = _as_parameter_list(parameters)
    g_i = collect_task_gradients(task_losses, params)
    T = g_i.size(0)
    pi = sample_projection_order(T, g_i.device)

    if cfg.implementation == "optimized":
        g_i_pc = pcgrad_project(g_i, pi, stability_eps=cfg.stability_eps)
    elif cfg.implementation == "reference":
        g_i_pc = pcgrad_project_reference(g_i, pi, stability_eps=cfg.stability_eps)
    else:
        raise ValueError("implementation must be 'optimized' or 'reference'")

    g = aggregate_projected_gradients(g_i_pc, reduction=cfg.reduction)
    loss = build_gradient_surrogate_loss(g, params)
    return GradientSurgeryProjectionOutput(loss=loss, g=g, g_i=g_i, g_i_pc=g_i_pc, pi=pi)
