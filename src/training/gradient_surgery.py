"""Gradient surgery: conflict detection, projection, and multi-task gradient aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------


def flatten_gradients(model: nn.Module) -> Tensor:
    """Concatenate all parameter gradients into one flat vector.

    Skips parameters with None gradient.

    Returns:
        Tensor of shape (total_params,).
    """
    parts: list[Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().flatten())
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def unflatten_gradients(flat_grad: Tensor, model: nn.Module) -> None:
    """Scatter flat_grad back into model.parameters() .grad fields.

    Only updates parameters that had non-None gradients (same skip logic as
    flatten_gradients).

    Args:
        flat_grad: flat gradient tensor produced by flatten_gradients.
        model: the model whose .grad fields will be updated.
    """
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.numel()
            p.grad = flat_grad[offset : offset + numel].reshape(p.shape).clone()
            offset += numel


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def compute_gradient_conflict(grad1: Tensor, grad2: Tensor) -> float:
    """Cosine similarity between two gradient vectors.

    Returns:
        float in [-1, 1]. Negative values indicate conflicting gradients.
    """
    g1 = grad1.flatten().float()
    g2 = grad2.flatten().float()
    norm1 = g1.norm()
    norm2 = g2.norm()
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return (torch.dot(g1, g2) / (norm1 * norm2)).item()


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_gradient(grad: Tensor, onto: Tensor) -> Tensor:
    """Project grad onto the direction of `onto`.

    Component: (grad · onto) / (onto · onto + 1e-8) * onto

    Returns:
        Tensor of the same shape as grad.
    """
    g = grad.flatten().float()
    o = onto.flatten().float()
    scale = torch.dot(g, o) / (torch.dot(o, o) + 1e-8)
    projection = scale * o
    return projection.reshape(grad.shape).to(grad.dtype)


# ---------------------------------------------------------------------------
# PCGrad aggregation
# ---------------------------------------------------------------------------


def gradient_surgery_step(gradients: list[Tensor]) -> Tensor:
    """PCGrad: for each pair of task gradients, if cosine similarity < 0,
    subtract the conflicting projection.

    For each task i and each other task j:
        if cos(g_i, g_j) < 0:
            g_i = g_i - proj(g_i, g_j)

    Args:
        gradients: list of flat gradient tensors, one per task.

    Returns:
        Mean of modified gradients, shape (total_params,).
    """
    n = len(gradients)
    if n == 0:
        return torch.zeros(0)

    modified = [g.clone().float() for g in gradients]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g_j = gradients[j].float()  # use original for reference
            dot = torch.dot(modified[i], g_j)
            if dot < 0:
                norm_sq = torch.dot(g_j, g_j)
                if norm_sq > 1e-12:
                    modified[i] = modified[i] - (dot / norm_sq) * g_j

    stacked = torch.stack(modified)
    return stacked.mean(dim=0).to(gradients[0].dtype)


# ---------------------------------------------------------------------------
# Gradient Vaccine
# ---------------------------------------------------------------------------


def gradient_vaccine(gradients: list[Tensor], epsilon: float = 0.1) -> Tensor:
    """Gradient Vaccine (Yu et al.): modify gradients to reduce inner-product variance.

    For each gradient g_i:
        g_avg = mean of all gradients
        g_i_modified = g_i + epsilon * (g_avg - proj(g_i, g_avg))

    Returns:
        Average of modified gradients, shape (total_params,).
    """
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


# ---------------------------------------------------------------------------
# GradientSurgeon
# ---------------------------------------------------------------------------


class GradientSurgeon:
    """Aggregates task gradients using various surgery methods.

    Args:
        method: one of "pcgrad" | "vaccine" | "mean".
    """

    def __init__(self, method: str = "pcgrad") -> None:
        if method not in ("pcgrad", "vaccine", "mean"):
            raise ValueError(f"Unknown method '{method}'. Choose from: pcgrad, vaccine, mean")
        self.method = method

    def aggregate(self, task_gradients: list[Tensor]) -> Tensor:
        """Aggregate a list of flat task gradient tensors into one.

        Args:
            task_gradients: list of flat gradient tensors, one per task.

        Returns:
            Aggregated flat gradient tensor.
        """
        if not task_gradients:
            return torch.zeros(0)

        if self.method == "pcgrad":
            return gradient_surgery_step(task_gradients)
        elif self.method == "vaccine":
            return gradient_vaccine(task_gradients)
        else:  # "mean"
            stacked = torch.stack([g.float() for g in task_gradients])
            return stacked.mean(dim=0).to(task_gradients[0].dtype)

    def conflict_matrix(self, task_gradients: list[Tensor]) -> Tensor:
        """Compute pairwise cosine similarities between task gradients.

        Args:
            task_gradients: list of flat gradient tensors, one per task.

        Returns:
            Tensor of shape (n_tasks, n_tasks).
        """
        n = len(task_gradients)
        matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = compute_gradient_conflict(task_gradients[i], task_gradients[j])
        return matrix


# ---------------------------------------------------------------------------
# PCGradProjector — core PCGrad projection logic (Yu et al., arXiv:2001.06782)
# ---------------------------------------------------------------------------


class PCGradProjector:
    """PCGrad (Project Conflicting Gradients) core projection.

    For each pair of tasks (i, j): if g_i · g_j < 0 (conflicting), project
        g_i ← g_i − (g_i · g_j / ‖g_j‖²) * g_j
    All conflicting pairs are processed; projected gradients are returned.
    """

    def project(self, grads: list[Tensor]) -> list[Tensor]:
        """Apply PCGrad projection to a list of gradient tensors.

        Args:
            grads: N gradient tensors, all the same shape (one per task).

        Returns:
            List of N projected gradient tensors, same shape as inputs.
        """
        n = len(grads)
        if n == 0:
            return []

        # Work in float32 to avoid precision issues; preserve original dtype.
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
        """Apply PCGrad per-parameter and write summed result to param.grad.

        Args:
            per_task_grads: N lists, each containing one gradient tensor per
                parameter in ``params``.  Shape: [n_tasks][n_params].
            params: the model parameters whose .grad fields will be written.
        """
        n_tasks = len(per_task_grads)
        n_params = len(params)

        for p_idx in range(n_params):
            task_grads_for_param = [per_task_grads[t][p_idx] for t in range(n_tasks)]
            projected = self.project(task_grads_for_param)
            summed = torch.stack([p.float() for p in projected]).sum(dim=0)
            params[p_idx].grad = summed.to(task_grads_for_param[0].dtype).reshape(
                params[p_idx].shape
            )


# ---------------------------------------------------------------------------
# PCGradOptimizer — wraps any optimizer with PCGrad
# ---------------------------------------------------------------------------


class PCGradOptimizer:
    """Wraps any PyTorch optimizer with PCGrad gradient surgery.

    Args:
        base_optimizer: any ``torch.optim.Optimizer`` instance.
        projector: a ``PCGradProjector`` instance.
    """

    def __init__(self, base_optimizer: Optimizer, projector: PCGradProjector) -> None:
        self.base_optimizer = base_optimizer
        self.projector = projector

    def zero_grad(self) -> None:
        """Zero gradients on the wrapped optimizer's parameter groups."""
        self.base_optimizer.zero_grad()

    def pc_step(self, per_task_losses: list[Tensor], params: list[Tensor]) -> None:
        """Compute per-task gradients, apply PCGrad, then call optimizer.step().

        Args:
            per_task_losses: one scalar loss Tensor per task.
            params: list of model parameters (must require_grad).
        """
        n_tasks = len(per_task_losses)
        per_task_grads: list[list[Tensor]] = []

        for t in range(n_tasks):
            # Zero out any accumulated grads before computing this task's grads.
            self.base_optimizer.zero_grad()
            per_task_losses[t].backward(retain_graph=(t < n_tasks - 1))
            task_grads = []
            for p in params:
                if p.grad is not None:
                    task_grads.append(p.grad.clone())
                else:
                    task_grads.append(torch.zeros_like(p))
            per_task_grads.append(task_grads)

        # Zero grads so we can write the PCGrad result cleanly.
        self.base_optimizer.zero_grad()
        self.projector.project_params(per_task_grads, params)
        self.base_optimizer.step()


# ---------------------------------------------------------------------------
# TaskLossAggregator — simple multi-task loss combiner (for comparison)
# ---------------------------------------------------------------------------


class TaskLossAggregator:
    """Combines multiple task losses into a single scalar for back-propagation.

    Args:
        weights: optional per-task weights.  If ``None``, uniform weights are
            used (equivalent to a simple mean).
    """

    def __init__(self, weights: list[float] | None = None) -> None:
        self.weights = weights

    def aggregate(self, losses: list[Tensor]) -> Tensor:
        """Return a weighted sum of task losses.

        Args:
            losses: list of scalar loss tensors, one per task.

        Returns:
            Scalar tensor representing the combined loss.
        """
        n = len(losses)
        if self.weights is None:
            w = [1.0 / n] * n
        else:
            w = self.weights
        total = sum(wi * li for wi, li in zip(w, losses))
        return total  # type: ignore[return-value]

    def gradient_conflict_ratio(self, per_task_grads: list[Tensor]) -> float:
        """Fraction of task pairs (i, j) with i < j where cos(g_i, g_j) < 0.

        Args:
            per_task_grads: list of gradient tensors (one per task).

        Returns:
            Float in [0, 1].
        """
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
