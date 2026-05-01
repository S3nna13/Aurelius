"""Gradient surgery: conflict detection, projection, and multi-task gradient aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def flatten_gradients(model: nn.Module) -> Tensor:
    """Concatenate all parameter gradients into one flat vector."""
    parts: list[Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().flatten())
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def unflatten_gradients(flat_grad: Tensor, model: nn.Module) -> None:
    """Scatter flat_grad back into model.parameters() .grad fields."""
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.numel()
            p.grad = flat_grad[offset : offset + numel].reshape(p.shape).clone()
            offset += numel


def compute_gradient_conflict(grad1: Tensor, grad2: Tensor) -> float:
    """Cosine similarity between two gradient vectors."""
    g1 = grad1.flatten().float()
    g2 = grad2.flatten().float()
    norm1 = g1.norm()
    norm2 = g2.norm()
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return (torch.dot(g1, g2) / (norm1 * norm2)).item()


def project_gradient(grad: Tensor, onto: Tensor) -> Tensor:
    """Project grad onto the direction of `onto`."""
    g = grad.flatten().float()
    o = onto.flatten().float()
    scale = torch.dot(g, o) / (torch.dot(o, o) + 1e-8)
    projection = scale * o
    return projection.reshape(grad.shape).to(grad.dtype)


def gradient_surgery_step(gradients: list[Tensor]) -> Tensor:
    """PCGrad-style surgery over a list of task gradient vectors."""
    n = len(gradients)
    if n == 0:
        return torch.zeros(0)

    modified = [g.clone().float() for g in gradients]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g_j = gradients[j].float()
            dot = torch.dot(modified[i], g_j)
            if dot < 0:
                norm_sq = torch.dot(g_j, g_j)
                if norm_sq > 1e-12:
                    modified[i] = modified[i] - (dot / norm_sq) * g_j

    stacked = torch.stack(modified)
    return stacked.mean(dim=0).to(gradients[0].dtype)


def gradient_vaccine(gradients: list[Tensor], epsilon: float = 0.1) -> Tensor:
    """Gradient Vaccine (Yu et al.) aggregation."""
    n = len(gradients)
    if n == 0:
        return torch.zeros(0)

    grads_f = [g.float() for g in gradients]
    g_avg = torch.stack(grads_f).mean(dim=0)

    modified: list[Tensor] = []
    for g_i in grads_f:
        proj = project_gradient(g_i, g_avg)
        g_mod = g_i + epsilon * (g_avg - proj)
        modified.append(g_mod)

    result = torch.stack(modified).mean(dim=0)
    return result.to(gradients[0].dtype)


class GradientSurgeon:
    """Aggregates task gradients using various surgery methods.

    The class supports the legacy aggregate/conflict-matrix API and the newer
    resolve() entrypoint used by the DAIES cycle tests.
    """

    def __init__(self, method: str = "pcgrad", threshold: float = 0.5) -> None:
        if method not in ("pcgrad", "vaccine", "mean"):
            raise ValueError(f"Unknown method '{method}'. Choose from: pcgrad, vaccine, mean")
        self.method = method
        self.threshold = threshold
        self._conflicts: int = 0
        self._total: int = 0

    def aggregate(self, task_gradients: list[Tensor]) -> Tensor:
        """Aggregate a list of flat task gradient tensors into one."""
        if not task_gradients:
            return torch.zeros(0)

        if self.method == "pcgrad":
            return gradient_surgery_step(task_gradients)
        if self.method == "vaccine":
            return gradient_vaccine(task_gradients)
        stacked = torch.stack([g.float() for g in task_gradients])
        return stacked.mean(dim=0).to(task_gradients[0].dtype)

    def conflict_matrix(self, task_gradients: list[Tensor]) -> Tensor:
        """Compute pairwise cosine similarities between task gradients."""
        n = len(task_gradients)
        matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = compute_gradient_conflict(task_gradients[i], task_gradients[j])
        return matrix

    def resolve(self, losses: list[Tensor], model: nn.Module) -> None:
        """Compute gradients for several losses and write a conflict-resolved result."""
        params = list(model.parameters())
        if not losses:
            model.zero_grad()
            return

        per_task_grads: list[list[Tensor]] = []
        flat_task_grads: list[Tensor] = []

        for task_idx, loss in enumerate(losses):
            model.zero_grad()
            loss.backward(retain_graph=task_idx < len(losses) - 1)

            task_grads: list[Tensor] = []
            flat_parts: list[Tensor] = []
            for p in params:
                if p.grad is not None:
                    grad = p.grad.clone()
                else:
                    grad = torch.zeros_like(p)
                task_grads.append(grad)
                flat_parts.append(grad.detach().flatten())

            per_task_grads.append(task_grads)
            flat_task_grads.append(torch.cat(flat_parts) if flat_parts else torch.zeros(0))

        n = len(flat_task_grads)
        for i in range(n):
            for j in range(i + 1, n):
                self._total += 1
                if compute_gradient_conflict(flat_task_grads[i], flat_task_grads[j]) < 0:
                    self._conflicts += 1

        projector = PCGradProjector()
        model.zero_grad()
        projector.project_params(per_task_grads, params)

    @property
    def conflict_rate(self) -> float:
        return self._conflicts / max(self._total, 1)


class PCGradProjector:
    """PCGrad (Project Conflicting Gradients) core projection."""

    def project(self, grads: list[Tensor]) -> list[Tensor]:
        """Apply PCGrad projection to a list of gradient tensors."""
        n = len(grads)
        if n == 0:
            return []

        orig_dtype = grads[0].dtype
        projected = [g.clone().float().flatten() for g in grads]
        originals = [g.float().flatten() for g in grads]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                g_j = originals[j]
                dot_ij = torch.dot(projected[i], g_j)
                if dot_ij < 0:
                    norm_sq_j = torch.dot(g_j, g_j)
                    if norm_sq_j > 1e-12:
                        projected[i] = projected[i] - (dot_ij / norm_sq_j) * g_j

        return [p.reshape(grads[k].shape).to(orig_dtype) for k, p in enumerate(projected)]

    def project_params(
        self,
        per_task_grads: list[list[Tensor]],
        params: list[Tensor],
    ) -> None:
        """Apply PCGrad per-parameter and write summed result to param.grad."""
        n_tasks = len(per_task_grads)
        n_params = len(params)

        for p_idx in range(n_params):
            task_grads_for_param = [per_task_grads[t][p_idx] for t in range(n_tasks)]
            projected = self.project(task_grads_for_param)
            summed = torch.stack([p.float() for p in projected]).sum(dim=0)
            params[p_idx].grad = summed.to(task_grads_for_param[0].dtype).reshape(
                params[p_idx].shape
            )


class PCGradOptimizer:
    """Wraps any PyTorch optimizer with PCGrad gradient surgery."""

    def __init__(self, base_optimizer: Optimizer, projector: PCGradProjector) -> None:
        self.base_optimizer = base_optimizer
        self.projector = projector

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def pc_step(self, per_task_losses: list[Tensor], params: list[Tensor]) -> None:
        n_tasks = len(per_task_losses)
        per_task_grads: list[list[Tensor]] = []

        for t in range(n_tasks):
            self.base_optimizer.zero_grad()
            per_task_losses[t].backward(retain_graph=(t < n_tasks - 1))
            task_grads = []
            for p in params:
                if p.grad is not None:
                    task_grads.append(p.grad.clone())
                else:
                    task_grads.append(torch.zeros_like(p))
            per_task_grads.append(task_grads)

        self.base_optimizer.zero_grad()
        self.projector.project_params(per_task_grads, params)
        self.base_optimizer.step()


class TaskLossAggregator:
    """Combines multiple task losses into a single scalar for back-propagation."""

    def __init__(self, weights: list[float] | None = None) -> None:
        self.weights = weights

    def aggregate(self, losses: list[Tensor]) -> Tensor:
        n = len(losses)
        if self.weights is None:
            w = [1.0 / n] * n
        else:
            w = self.weights
        total = sum(wi * li for wi, li in zip(w, losses))
        return total  # type: ignore[return-value]

    def gradient_conflict_ratio(self, per_task_grads: list[Tensor]) -> float:
        n = len(per_task_grads)
        if n < 2:
            return 0.0
        n_pairs = n * (n - 1) // 2
        n_conflicting = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = compute_gradient_conflict(per_task_grads[i], per_task_grads[j])
                if cos_sim < 0:
                    n_conflicting += 1
        return n_conflicting / n_pairs
