"""Multi-task learning v2: MTLConfig, TaskHead, MTLLoss, MultiTaskModel, GradientBalancer.

New API on top of the existing multitask_learning.py backbone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MTLConfig:
    task_names: List[str] = field(default_factory=lambda: ["task_a", "task_b"])
    loss_weights: Optional[Dict[str, float]] = None
    gradient_accumulation_steps: int = 1
    uncertainty_weighting: bool = False


class TaskHead(nn.Module):
    """Task-specific projection head."""

    def __init__(self, d_model: int, output_dim: int, task_name: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.linear = nn.Linear(d_model, output_dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden)


def compute_uncertainty_weights(log_vars: Tensor) -> Tensor:
    """Uncertainty-based weighting: w_i = exp(-log_var_i).

    Args:
        log_vars: (n_tasks,) learnable log variances.

    Returns:
        (n_tasks,) positive weights.
    """
    return torch.exp(-log_vars)


def compute_gradient_cosine_similarity(grad_a: Tensor, grad_b: Tensor) -> float:
    """Cosine similarity between two gradient vectors."""
    a = grad_a.flatten().float()
    b = grad_b.flatten().float()
    denom = a.norm() * b.norm()
    if denom < 1e-8:
        return 0.0
    return (a @ b / denom).item()


class MTLLoss:
    """Weighted multi-task loss, optionally with uncertainty weighting."""

    def __init__(self, config: MTLConfig) -> None:
        self.config = config
        self.n_tasks = len(config.task_names)
        if config.uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        else:
            self.log_vars = None

    def compute(
        self, task_losses: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute weighted total loss.

        Returns:
            (total_loss_scalar, per_task_weight_dict)
        """
        names = self.config.task_names
        losses = [task_losses[n] for n in names if n in task_losses]
        active_names = [n for n in names if n in task_losses]

        if self.config.uncertainty_weighting and self.log_vars is not None:
            weights = compute_uncertainty_weights(self.log_vars[: len(active_names)])
        elif self.config.loss_weights is not None:
            weights = torch.tensor(
                [self.config.loss_weights.get(n, 1.0) for n in active_names],
                dtype=torch.float32,
            )
        else:
            weights = torch.ones(len(active_names), dtype=torch.float32)

        total = sum(w * l for w, l in zip(weights, losses))
        weight_dict = {n: w.item() for n, w in zip(active_names, weights)}
        return total, weight_dict


class MultiTaskModel(nn.Module):
    """Shared backbone + task-specific heads."""

    def __init__(
        self,
        backbone: nn.Module,
        task_heads: Dict[str, TaskHead],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(
        self, x: Tensor, task_name: Optional[str] = None
    ) -> Dict[str, Tensor]:
        hidden = self.backbone(x)
        if task_name is not None:
            return {task_name: self.task_heads[task_name](hidden)}
        return {name: head(hidden) for name, head in self.task_heads.items()}


class GradientBalancer:
    """Compute per-task gradient norms and inverse-norm balancing weights."""

    def __init__(self, n_tasks: int) -> None:
        self.n_tasks = n_tasks

    def compute_grad_norms(
        self,
        model: nn.Module,
        task_losses: List[Tensor],
        retain_graph: bool = True,
    ) -> List[float]:
        """Compute gradient norm for each task loss independently."""
        params = list(model.parameters())
        norms = []
        for i, loss in enumerate(task_losses):
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=(retain_graph or i < len(task_losses) - 1),
                allow_unused=True,
            )
            norm = sum(
                g.norm().item() ** 2 for g in grads if g is not None
            ) ** 0.5
            norms.append(norm)
        return norms

    def balance_weights(self, grad_norms: List[float]) -> List[float]:
        """Inverse-norm weights normalized to sum to 1."""
        eps = 1e-8
        inv_norms = [1.0 / (n + eps) for n in grad_norms]
        total = sum(inv_norms)
        return [w / total for w in inv_norms]
