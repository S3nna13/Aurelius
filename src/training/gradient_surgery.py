"""PCGrad: Gradient Surgery for multi-task learning.

Resolves gradient conflicts between tasks by projecting conflicting gradients
onto each other's normal planes.

Reference: Yu et al., 2020 - "Gradient Surgery for Multi-Task Learning"
           arXiv:2001.06782
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable


class PCGrad:
    """
    PCGrad: Projecting Conflicting Gradients.

    For each pair of task gradients (g_i, g_j):
    If cos(g_i, g_j) < 0 (conflicting):
        g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j

    Args:
        optimizer: underlying optimizer
        reduction: 'sum' | 'mean' for combining task gradients
    """

    def __init__(self, optimizer: torch.optim.Optimizer, reduction: str = 'mean') -> None:
        self.optimizer = optimizer
        self.reduction = reduction
        self._params: list[torch.nn.Parameter] = []
        for group in optimizer.param_groups:
            self._params.extend(group['params'])

    def zero_grad(self) -> None:
        """Zero gradients on the underlying optimizer."""
        self.optimizer.zero_grad()

    def pc_backward(self, losses: list[torch.Tensor]) -> None:
        """
        Compute PCGrad update for a list of task losses.

        1. For each task i: compute gradient g_i = grad(loss_i)
        2. For each task i, for each other task j:
           - If g_i . g_j < 0: project g_i away from g_j
        3. Sum (or mean) projected gradients across tasks
        4. Set parameter .grad to projected sum/mean

        Args:
            losses: list of per-task scalar losses (one per task)
        """
        # Collect per-task flat gradients
        task_grads: list[list[torch.Tensor]] = []
        for loss in losses:
            # Zero before each backward so gradients don't accumulate
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # Snapshot per-parameter gradients for this task
            per_param = []
            for p in self._params:
                if p.grad is not None:
                    per_param.append(p.grad.detach().clone())
                else:
                    per_param.append(torch.zeros_like(p))
            task_grads.append(per_param)

        n_tasks = len(task_grads)
        n_params = len(self._params)

        # Apply PCGrad per parameter across tasks
        projected: list[list[torch.Tensor]] = []
        for i in range(n_tasks):
            proj_i = [g.clone() for g in task_grads[i]]
            for j in range(n_tasks):
                if i == j:
                    continue
                # Project each parameter gradient independently
                new_proj_i = []
                for k in range(n_params):
                    g_i_k = proj_i[k].view(-1)
                    g_j_k = task_grads[j][k].view(-1)
                    g_i_k_proj = self.project_gradient(g_i_k, g_j_k)
                    new_proj_i.append(g_i_k_proj.view_as(proj_i[k]))
                proj_i = new_proj_i
            projected.append(proj_i)

        # Combine projected gradients and assign to .grad
        self.optimizer.zero_grad()
        for k, p in enumerate(self._params):
            param_grads = torch.stack([projected[i][k] for i in range(n_tasks)])
            if self.reduction == 'mean':
                p.grad = param_grads.mean(dim=0)
            else:  # 'sum'
                p.grad = param_grads.sum(dim=0)

    def step(self) -> None:
        """Take an optimizer step."""
        self.optimizer.step()

    @staticmethod
    def project_gradient(g_i: torch.Tensor, g_j: torch.Tensor) -> torch.Tensor:
        """
        Project g_i to remove components conflicting with g_j.

        If g_i . g_j >= 0: no conflict, return g_i unchanged.
        If g_i . g_j < 0: return g_i - (g_i.g_j / ||g_j||^2) * g_j

        Args:
            g_i: flat gradient tensor to (potentially) project
            g_j: flat reference gradient tensor

        Returns:
            Projected gradient (same shape as g_i).
        """
        dot = torch.dot(g_i.view(-1), g_j.view(-1))
        if dot < 0:
            g_j_norm_sq = torch.dot(g_j.view(-1), g_j.view(-1))
            g_i = g_i - (dot / (g_j_norm_sq + 1e-12)) * g_j
        return g_i


class GradientSurgeryMonitor:
    """Monitor gradient conflicts during training."""

    def __init__(self) -> None:
        self.conflict_history: list[float] = []

    def measure_conflict(self, gradients: list[torch.Tensor]) -> float:
        """
        Measure fraction of (i, j) gradient pairs that conflict (cos < 0).

        Args:
            gradients: list of flat gradient tensors, one per task.

        Returns:
            float in [0, 1] representing the fraction of conflicting pairs.
        """
        n = len(gradients)
        if n < 2:
            return 0.0

        total_pairs = 0
        conflicting_pairs = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                g_i = gradients[i].view(-1)
                g_j = gradients[j].view(-1)
                dot = torch.dot(g_i, g_j).item()
                total_pairs += 1
                if dot < 0:
                    conflicting_pairs += 1

        if total_pairs == 0:
            return 0.0

        return conflicting_pairs / total_pairs

    def record(self, gradients: list[torch.Tensor]) -> None:
        """Measure and record conflict fraction."""
        rate = self.measure_conflict(gradients)
        self.conflict_history.append(rate)

    def mean_conflict_rate(self) -> float:
        """Mean conflict rate across recorded steps."""
        if not self.conflict_history:
            return 0.0
        return sum(self.conflict_history) / len(self.conflict_history)


class MultiTaskTrainer:
    """
    Train a model on multiple tasks simultaneously using gradient surgery.

    Args:
        model: shared model
        task_loss_fns: list of callables, each takes (model, batch) -> scalar loss
        optimizer: shared optimizer
        use_pcgrad: whether to use PCGrad (True) or simple sum (False)
    """

    def __init__(
        self,
        model: nn.Module,
        task_loss_fns: list[Callable],
        optimizer: torch.optim.Optimizer,
        use_pcgrad: bool = True,
    ) -> None:
        self.model = model
        self.task_loss_fns = task_loss_fns
        self.optimizer = optimizer
        self.use_pcgrad = use_pcgrad
        self.monitor = GradientSurgeryMonitor()

        if use_pcgrad:
            self.pcgrad = PCGrad(optimizer, reduction='mean')
        else:
            self.pcgrad = None

    def train_step(self, batches: list) -> dict:
        """
        Compute losses for all tasks, apply PCGrad or simple backward.

        Args:
            batches: list of batches, one per task.

        Returns:
            {
                'task_losses': [float],
                'total_loss': float,
                'conflict_rate': float,
            }
        """
        # Compute per-task losses
        losses = []
        for fn, batch in zip(self.task_loss_fns, batches):
            loss = fn(self.model, batch)
            losses.append(loss)

        task_loss_values = [l.item() for l in losses]
        total_loss = sum(task_loss_values)

        if self.use_pcgrad and self.pcgrad is not None:
            # Collect pre-PCGrad gradients for conflict measurement
            raw_grads = []
            for loss in losses:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                flat = []
                for p in self.model.parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            flat.append(p.grad.detach().view(-1))
                        else:
                            flat.append(torch.zeros(p.numel()))
                raw_grads.append(torch.cat(flat))

            conflict_rate = self.monitor.measure_conflict(raw_grads)
            self.monitor.conflict_history.append(conflict_rate)

            # Apply PCGrad
            self.pcgrad.pc_backward(losses)
        else:
            # Simple sum of gradients
            self.optimizer.zero_grad()
            total = sum(losses)
            total.backward()

            # Measure conflict from individual task gradients
            raw_grads = []
            for loss in losses:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                flat = []
                for p in self.model.parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            flat.append(p.grad.detach().view(-1))
                        else:
                            flat.append(torch.zeros(p.numel()))
                raw_grads.append(torch.cat(flat))
            conflict_rate = self.monitor.measure_conflict(raw_grads)
            self.monitor.conflict_history.append(conflict_rate)

            # Recompute the actual backward for the step
            self.optimizer.zero_grad()
            sum(losses).backward()

        return {
            'task_losses': task_loss_values,
            'total_loss': total_loss,
            'conflict_rate': conflict_rate,
        }
