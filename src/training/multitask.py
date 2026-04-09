"""Multi-task learning: task-specific heads, dynamic loss balancing, and gradient conflict resolution."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskHead:
    """A task-specific head with its own loss function."""
    name: str
    head: nn.Module
    loss_fn: Callable
    weight: float = 1.0


@dataclass
class MultitaskConfig:
    """Configuration for multi-task training."""
    balancing_strategy: str = "static"          # "static" | "uncertainty" | "dynamic_temperature"
    gradient_surgery: bool = False              # PCGrad-style conflict resolution
    temperature_lr: float = 0.01               # learning rate for dynamic temperature params
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Gradient surgery
# ---------------------------------------------------------------------------

def project_conflicting_gradients(grads: list[Tensor]) -> list[Tensor]:
    """PCGrad: project out conflicting gradient components.

    For each pair (g_i, g_j), if cos(g_i, g_j) < 0:
        g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j

    Returns de-conflicted gradients (new tensors).
    """
    result = [g.clone() for g in grads]
    for i in range(len(result)):
        for j in range(len(result)):
            if i == j:
                continue
            g_j = grads[j]  # use original grads for projection targets
            dot = torch.dot(result[i].flatten(), g_j.flatten())
            if dot < 0:
                norm_sq = torch.dot(g_j.flatten(), g_j.flatten())
                if norm_sq > 0:
                    result[i] = result[i] - (dot / norm_sq) * g_j
    return result


# ---------------------------------------------------------------------------
# Loss balancing strategies
# ---------------------------------------------------------------------------

class UncertaintyWeighting(nn.Module):
    """Kendall et al. multi-task uncertainty weighting.

    weighted_loss = sum( exp(-log_var_i) * loss_i + log_var_i )
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[Tensor]) -> Tensor:
        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


class DynamicTemperatureBalancing(nn.Module):
    """Scale losses by learnable temperatures.

    total = sum( loss_i / temp_i + log(temp_i) )
    """

    def __init__(self, n_tasks: int, lr: float = 0.01) -> None:
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(n_tasks))
        self.lr = lr

    def forward(self, losses: list[Tensor]) -> Tensor:
        total = torch.tensor(0.0, device=self.temperatures.device)
        for i, loss in enumerate(losses):
            total = total + loss / self.temperatures[i] + torch.log(self.temperatures[i])
        return total


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------

class MultitaskModel(nn.Module):
    """Shared backbone with multiple task-specific heads."""

    def __init__(
        self,
        backbone: nn.Module,
        task_heads: list[TaskHead],
        config: MultitaskConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Store task heads as ModuleDict so their params are registered
        self._task_heads_meta: dict[str, TaskHead] = {}
        self.heads = nn.ModuleDict()
        for th in task_heads:
            self._task_heads_meta[th.name] = th
            self.heads[th.name] = th.head

    def forward(self, input_ids: Tensor, task_name: str) -> Tensor:
        """Run backbone then apply the named task head.

        Args:
            input_ids: (batch, seq_len) token indices.
            task_name: which task head to apply.

        Returns:
            Task head output tensor.
        """
        if task_name not in self._task_heads_meta:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(self._task_heads_meta)}"
            )

        # Backbone returns (loss, logits, past_key_values)
        _loss, logits, _pkv = self.backbone(input_ids)

        # Mean-pool logits over the time dimension: (B, T, V) -> (B, V)
        pooled = logits.mean(dim=1)

        # Apply task-specific head
        head = self.heads[task_name]
        return head(pooled)


# ---------------------------------------------------------------------------
# Multi-task trainer
# ---------------------------------------------------------------------------

class MultitaskTrainer:
    """Orchestrates multi-task training with loss balancing and optional gradient surgery."""

    def __init__(
        self,
        model: MultitaskModel,
        optimizer: torch.optim.Optimizer,
        config: MultitaskConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        n_tasks = len(model._task_heads_meta)

        # Set up balancing strategy
        if config.balancing_strategy == "uncertainty":
            self.balancer: UncertaintyWeighting | DynamicTemperatureBalancing | None = (
                UncertaintyWeighting(n_tasks)
            )
            # Add balancer params to optimizer
            self.optimizer.add_param_group({"params": self.balancer.parameters()})
        elif config.balancing_strategy == "dynamic_temperature":
            self.balancer = DynamicTemperatureBalancing(n_tasks, lr=config.temperature_lr)
            self.optimizer.add_param_group(
                {"params": self.balancer.parameters(), "lr": config.temperature_lr}
            )
        else:
            self.balancer = None

    def compute_all_losses(
        self, input_ids: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Forward each task and compute its loss.

        Returns dict mapping task_name -> scalar loss tensor.
        """
        losses: dict[str, Tensor] = {}
        for name, th in self.model._task_heads_meta.items():
            if name not in targets:
                continue
            output = self.model(input_ids, name)
            loss = th.loss_fn(output, targets[name])
            losses[name] = loss
        return losses

    def _balance_losses(self, losses: dict[str, Tensor]) -> Tensor:
        """Combine per-task losses according to the configured strategy."""
        task_names = list(self.model._task_heads_meta.keys())
        loss_list = [losses[n] for n in task_names if n in losses]

        if self.balancer is not None:
            return self.balancer(loss_list)

        # Static weighting
        total = torch.tensor(0.0, device=loss_list[0].device)
        for name in task_names:
            if name in losses:
                th = self.model._task_heads_meta[name]
                total = total + th.weight * losses[name]
        return total

    def train_step(
        self, input_ids: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, float]:
        """One training step: forward all tasks, balance, optional gradient surgery, backward, step.

        Returns dict with 'total_loss' and per-task losses as floats.
        """
        self.model.train()
        self.optimizer.zero_grad()

        losses = self.compute_all_losses(input_ids, targets)

        if self.config.gradient_surgery and len(losses) > 1:
            # Compute per-task gradients, then apply PCGrad
            task_grads: list[Tensor] = []
            for name, loss in losses.items():
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # Flatten all backbone param grads into one vector
                grad_vec = torch.cat([
                    p.grad.flatten() for p in self.model.backbone.parameters()
                    if p.grad is not None
                ])
                task_grads.append(grad_vec)

            # Project conflicting gradients
            deconflicted = project_conflicting_gradients(task_grads)
            avg_grad = torch.stack(deconflicted).mean(dim=0)

            # Write back averaged de-conflicted gradient
            self.optimizer.zero_grad()
            offset = 0
            for p in self.model.backbone.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                numel = p.numel()
                p.grad = avg_grad[offset:offset + numel].reshape(p.shape).clone()
                offset += numel

            # Also backward the balanced loss for task head grads
            total = self._balance_losses(losses)
            # We need head gradients -- recompute on heads only
            for name, loss in losses.items():
                loss.backward(retain_graph=True)
                # This adds to existing grads; backbone grads already set above
            # We already set backbone grads, so just keep head grads from the backward passes
        else:
            total = self._balance_losses(losses)
            total.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        self.optimizer.step()

        # Build result dict
        total_val = sum(
            self.model._task_heads_meta[n].weight * losses[n].item()
            for n in losses
        )
        result: dict[str, float] = {"total_loss": total_val}
        for name, loss in losses.items():
            result[f"{name}_loss"] = loss.item()

        return result
