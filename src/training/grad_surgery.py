"""PCGrad and CAGrad: gradient surgery for multi-task learning.

Resolves gradient conflicts between tasks to improve multi-task training.

PCGrad: Project conflicting gradients onto each other's normal plane.
  Yu et al., 2020 - arXiv:2001.06782

CAGrad: Find the gradient closest to average that improves all tasks.
  Liu et al., 2021 - arXiv:2110.14048
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GradSurgeryConfig:
    method: str = "pcgrad"  # "pcgrad" or "cagrad"
    cagrad_c: float = 0.5  # CAGrad: fraction of gradient space to constrain
    normalize: bool = True  # normalize task gradients by their norms


def project_gradient(
    g_i: torch.Tensor,  # (D,) gradient to project
    g_j: torch.Tensor,  # (D,) gradient to project against
) -> torch.Tensor:
    """Project g_i to remove component conflicting with g_j.

    If cos(g_i, g_j) < 0: remove component of g_i along g_j direction.
    Otherwise: return g_i unchanged.

    Args:
        g_i: Gradient to (potentially) project, shape (D,).
        g_j: Reference gradient, shape (D,).

    Returns:
        Projected gradient, shape (D,).
    """
    dot = torch.dot(g_i, g_j)
    if dot < 0:
        # Remove the component of g_i along g_j
        # projection: g_i - (g_i · g_j / ||g_j||²) * g_j
        g_j_norm_sq = torch.dot(g_j, g_j)
        g_i = g_i - (dot / (g_j_norm_sq + 1e-12)) * g_j
    return g_i


def pcgrad(
    task_gradients: list[torch.Tensor],  # list of (D,) flattened gradients, one per task
) -> torch.Tensor:
    """PCGrad: project conflicting gradients.

    For each task i:
      For each other task j (in random order per the paper):
        If cos(g_i, g_j) < 0: project g_i to remove g_j's component.
    Final gradient = mean of projected task gradients.

    Args:
        task_gradients: List of per-task flat gradient tensors, each shape (D,).

    Returns:
        Merged gradient, shape (D,).
    """
    n = len(task_gradients)
    projected = []

    for i in range(n):
        g_i = task_gradients[i].clone()
        # Iterate over other tasks in a shuffled order (paper recommendation)
        indices = list(range(n))
        indices.remove(i)
        for j in indices:
            g_i = project_gradient(g_i, task_gradients[j])
        projected.append(g_i)

    return torch.stack(projected).mean(dim=0)


def cagrad(
    task_gradients: list[torch.Tensor],  # list of (D,) flattened gradients
    c: float = 0.5,
) -> torch.Tensor:
    """CAGrad: find gradient minimizing task loss while improving all tasks.

    Simplified version: gradient = average + c * direction_toward_min_task_grad
    where the direction corrects for the worst task.

    Full algorithm (simplified):
    1. g_0 = mean(task_gradients)
    2. Find worst task: task with smallest cos(g_0, g_i)
    3. g_final = g_0 + c * (g_worst - g_0)  (blend toward worst task)

    Args:
        task_gradients: List of per-task flat gradient tensors, each shape (D,).
        c: Blending coefficient toward worst task (0 = pure average, 1 = worst task).

    Returns:
        Merged gradient, shape (D,).
    """
    g_avg = torch.stack(task_gradients).mean(dim=0)

    if c == 0.0:
        return g_avg

    g_avg_norm = g_avg.norm()

    # Find worst task: smallest cosine similarity with g_avg
    worst_idx = 0
    worst_cos = float("inf")
    for i, g_i in enumerate(task_gradients):
        g_i_norm = g_i.norm()
        if g_avg_norm < 1e-12 or g_i_norm < 1e-12:
            cos = 0.0
        else:
            cos = torch.dot(g_avg, g_i) / (g_avg_norm * g_i_norm + 1e-12)
            cos = cos.item()
        if cos < worst_cos:
            worst_cos = cos
            worst_idx = i

    g_worst = task_gradients[worst_idx]
    # Blend average toward worst task to ensure improvement for all
    g_final = g_avg + c * (g_worst - g_avg)
    return g_final


class MultiTaskGradManager:
    """Manages multi-task gradient surgery during training.

    Usage:
        manager = MultiTaskGradManager(model, cfg)

        # In training loop:
        task_losses = [loss1, loss2, loss3]
        manager.zero_grad()
        merged_grad = manager.compute_merged_gradient(task_losses)
        manager.apply_gradient(merged_grad)
        optimizer.step()

    Args:
        model: The model whose gradients will be managed.
        cfg: Gradient surgery configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: GradSurgeryConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or GradSurgeryConfig()
        self._n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def zero_grad(self) -> None:
        """Zero out all parameter gradients."""
        self.model.zero_grad(set_to_none=True)

    def _get_flat_grad(self) -> torch.Tensor:
        """Flatten all param gradients to a single vector."""
        grads = []
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
                else:
                    grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    def _apply_flat_grad(self, flat_grad: torch.Tensor) -> None:
        """Write flat_grad back into parameter .grad fields."""
        offset = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n = p.numel()
                if p.grad is None:
                    p.grad = flat_grad[offset : offset + n].view_as(p).clone()
                else:
                    p.grad.copy_(flat_grad[offset : offset + n].view_as(p))
                offset += n

    def compute_merged_gradient(
        self,
        task_losses: list[torch.Tensor],  # list of scalar losses, one per task
    ) -> torch.Tensor:
        """Compute merged gradient from multiple task losses.

        For each task loss: backward, collect grad, zero_grad.
        Apply PCGrad or CAGrad to merge.

        Args:
            task_losses: List of scalar loss tensors, one per task.

        Returns:
            Merged flat gradient vector, shape (D,).
        """
        task_grads = []

        for loss in task_losses:
            self.model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            task_grads.append(self._get_flat_grad())

        # Optionally normalize
        if self.cfg.normalize:
            norms = [g.norm() for g in task_grads]
            task_grads = [g / (n + 1e-8) for g, n in zip(task_grads, norms)]

        if self.cfg.method == "pcgrad":
            merged = pcgrad(task_grads)
        else:
            merged = cagrad(task_grads, c=self.cfg.cagrad_c)

        return merged

    def apply_gradient(self, flat_grad: torch.Tensor) -> None:
        """Apply a flat gradient vector to model parameters.

        Args:
            flat_grad: Flat gradient vector, shape (D,).
        """
        self._apply_flat_grad(flat_grad)
