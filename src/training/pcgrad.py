"""PCGrad: Gradient Surgery for Multi-Task Learning.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PCGradConfig:
    n_tasks: int = 2
    reduction: str = "mean"   # "mean" | "sum" — how to combine task losses


def project_gradient(grad_i: Tensor, grad_j: Tensor) -> Tensor:
    """Project grad_i onto normal plane of grad_j if they conflict.

    If dot(g_i, g_j) < 0: g_i = g_i - (dot(g_i, g_j) / ||g_j||^2) * g_j
    else: g_i unchanged.
    Both tensors are 1D (flattened).
    Returns projected grad_i.
    """
    dot = torch.dot(grad_i, grad_j)
    if dot < 0:
        norm_sq = torch.dot(grad_j, grad_j)
        if norm_sq > 0:
            grad_i = grad_i - (dot / norm_sq) * grad_j
    return grad_i


def pcgrad_step(
    losses: list[Tensor],
    params: list[nn.Parameter],
    retain_graph: bool = False,
) -> Tensor:
    """Apply PCGrad: compute per-task gradients, project, accumulate, apply.

    Steps:
    1. For each task loss, compute gradient w.r.t. each param
    2. For each task i, project g_i against all g_j (j != i)
    3. Sum the projected gradients into param.grad
    4. Return mean of task losses (scalar)

    Args:
        losses: list of per-task scalar loss tensors
        params: list of parameters to update
        retain_graph: whether to retain computation graph
    Returns:
        mean loss (scalar tensor)
    """
    n_tasks = len(losses)

    # Step 1: compute per-task per-param gradients
    # task_grads[i][k] = gradient of losses[i] w.r.t. params[k] (1D tensor or None)
    task_grads: list[list[Tensor | None]] = []
    for i, loss in enumerate(losses):
        # retain graph for all but the last task
        keep = retain_graph or (i < n_tasks - 1)
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=keep,
            create_graph=False,
            allow_unused=True,
        )
        task_grads.append(list(grads))

    # Step 2 & 3: for each param, project and accumulate
    for k, param in enumerate(params):
        # Collect flat gradients for this param across tasks
        flat_grads: list[Tensor | None] = []
        for i in range(n_tasks):
            g = task_grads[i][k]
            if g is not None:
                flat_grads.append(g.flatten())
            else:
                flat_grads.append(None)

        # Project each task's gradient against all others
        projected: list[Tensor] = []
        for i in range(n_tasks):
            if flat_grads[i] is None:
                continue
            g_i = flat_grads[i].clone()
            for j in range(n_tasks):
                if i == j or flat_grads[j] is None:
                    continue
                g_i = project_gradient(g_i, flat_grads[j])
            projected.append(g_i)

        if not projected:
            continue

        # Sum projected gradients and assign to param.grad
        combined = torch.stack(projected).sum(dim=0)
        param.grad = combined.view_as(param).clone()

    # Step 4: return mean loss
    stacked = torch.stack([l.detach() for l in losses])
    return stacked.mean()


class PCGradOptimizer:
    """Wrapper that applies PCGrad logic before optimizer.step()."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: PCGradConfig) -> None:
        self.optimizer = optimizer
        self.config = config

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, task_losses: list[Tensor]) -> dict:
        """Compute PCGrad gradients and take optimizer step.

        Returns dict with keys:
            "mean_loss": float
            "n_conflicts": int  — total gradient conflicts detected
            "task_losses": list[float]
        """
        # Gather all parameters from optimizer param groups
        params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

        n_tasks = len(task_losses)
        n_conflicts = 0

        # Compute per-task per-param gradients
        task_grads: list[list[Tensor | None]] = []
        for i, loss in enumerate(task_losses):
            keep = i < n_tasks - 1
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=keep,
                create_graph=False,
                allow_unused=True,
            )
            task_grads.append(list(grads))

        # Project and accumulate per param
        for k, param in enumerate(params):
            flat_grads: list[Tensor | None] = []
            for i in range(n_tasks):
                g = task_grads[i][k]
                flat_grads.append(g.flatten() if g is not None else None)

            projected: list[Tensor] = []
            for i in range(n_tasks):
                if flat_grads[i] is None:
                    continue
                g_i = flat_grads[i].clone()
                for j in range(n_tasks):
                    if i == j or flat_grads[j] is None:
                        continue
                    dot = torch.dot(g_i, flat_grads[j])
                    if dot < 0:
                        n_conflicts += 1
                        norm_sq = torch.dot(flat_grads[j], flat_grads[j])
                        if norm_sq > 0:
                            g_i = g_i - (dot / norm_sq) * flat_grads[j]
                projected.append(g_i)

            if not projected:
                continue

            combined = torch.stack(projected).sum(dim=0)
            param.grad = combined.view_as(param).clone()

        self.optimizer.step()

        return {
            "mean_loss": float(torch.stack([l.detach() for l in task_losses]).mean()),
            "n_conflicts": n_conflicts,
            "task_losses": [float(l.detach()) for l in task_losses],
        }


class MultiTaskPCGradTrainer:
    """Trains model on multiple tasks using PCGrad."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: PCGradConfig,
        task_fns: list[Callable[[nn.Module], Tensor]],
    ) -> None:
        self.model = model
        self.config = config
        self.task_fns = task_fns
        self.pcgrad_optimizer = PCGradOptimizer(optimizer, config)

    def train_step(self) -> dict:
        """Run one PCGrad step across all tasks.

        Returns dict with loss, n_conflicts, task_losses.
        """
        self.model.train()
        self.pcgrad_optimizer.zero_grad()

        task_losses = [fn(self.model) for fn in self.task_fns]
        result = self.pcgrad_optimizer.step(task_losses)

        return {
            "loss": result["mean_loss"],
            "n_conflicts": result["n_conflicts"],
            "task_losses": result["task_losses"],
        }
