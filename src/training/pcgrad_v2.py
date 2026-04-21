"""
PCGrad v2: Projecting Conflicting Gradients with Cosine Adaptive Weighting (2025).

Improvements over PCGrad (2020):
  1. Cosine-similarity-based adaptive weighting (instead of binary conflict detection).
  2. Per-layer conflict tracking.
  3. A "gradient bank" that accumulates gradients across mini-batches before resolving.

Algorithm:
  For tasks A and B with gradients g_A and g_B:
    cos_neg = max(0, -cos(g_A, g_B))
    g_A_resolved = g_A - alpha * cos_neg * (g_A·g_B / ||g_B||²) * g_B
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PCGradV2Config:
    """Configuration for PCGrad v2."""
    n_tasks: int = 2
    adaptive_weight: bool = True   # use cosine-adaptive weighting
    alpha: float = 1.0             # weighting factor for projection
    normalize_gradients: bool = False  # normalize grad magnitudes before conflict check
    gradient_bank_size: int = 1    # accumulate this many batches before resolving


# ---------------------------------------------------------------------------
# Gradient Bank
# ---------------------------------------------------------------------------

class GradientBank:
    """Accumulates per-task gradients for multi-batch conflict resolution."""

    def __init__(self, n_tasks: int, bank_size: int) -> None:
        self.n_tasks = n_tasks
        self.bank_size = bank_size
        # _storage[task_id] -> list of batch grad-lists
        self._storage: dict[int, list[list[Tensor]]] = {
            t: [] for t in range(n_tasks)
        }

    def add(self, task_id: int, grads: list[Tensor]) -> bool:
        """Add one batch of gradients for the given task.

        Returns True when *all* tasks have accumulated ``bank_size`` batches
        (i.e. the bank is full and ready to resolve).
        """
        if task_id not in self._storage:
            self._storage[task_id] = []
        self._storage[task_id].append([g.clone() for g in grads])

        # Full when every task has >= bank_size entries
        return all(
            len(self._storage[t]) >= self.bank_size for t in range(self.n_tasks)
        )

    def get_accumulated(self) -> list[list[Tensor]]:
        """Return accumulated (averaged) gradients: [n_tasks][n_params]."""
        result: list[list[Tensor]] = []
        for t in range(self.n_tasks):
            batches = self._storage[t][: self.bank_size]  # use first bank_size batches
            if not batches:
                raise RuntimeError(f"No gradients stored for task {t}")
            n_params = len(batches[0])
            # Average across accumulated batches
            avg_grads: list[Tensor] = []
            for p_idx in range(n_params):
                stacked = torch.stack([b[p_idx] for b in batches], dim=0)
                avg_grads.append(stacked.mean(dim=0))
            result.append(avg_grads)
        return result

    def clear(self) -> None:
        """Reset the bank."""
        self._storage = {t: [] for t in range(self.n_tasks)}


# ---------------------------------------------------------------------------
# PCGrad v2 Core
# ---------------------------------------------------------------------------

class PCGradV2:
    """PCGrad v2 — Cosine-Adaptive Conflicting Gradient Projection."""

    def __init__(self, config: Optional[PCGradV2Config] = None) -> None:
        self.config = config if config is not None else PCGradV2Config()

    # ------------------------------------------------------------------
    # Primitive operations
    # ------------------------------------------------------------------

    def conflict_score(self, g1: Tensor, g2: Tensor) -> float:
        """Cosine similarity between g1 and g2.

        Returns a float in [-1, 1].  Values < 0 indicate a conflict.
        """
        g1_flat = g1.reshape(-1).float()
        g2_flat = g2.reshape(-1).float()
        denom = g1_flat.norm() * g2_flat.norm() + 1e-8
        return (g1_flat @ g2_flat / denom).item()

    def project(self, g1: Tensor, g2: Tensor) -> Tensor:
        """Resolve conflict in g1 with respect to g2.

        g1_resolved = g1 - alpha * cos_neg * (g1·g2 / ||g2||²) * g2
        where  cos_neg = max(0, -cos(g1, g2)).

        If adaptive_weight is False, cos_neg is replaced by a binary indicator
        (1 if cos < 0 else 0), replicating original PCGrad behaviour with the
        alpha scaling.
        """
        cos = self.conflict_score(g1, g2)

        if self.config.adaptive_weight:
            weight = max(0.0, -cos)
        else:
            weight = 1.0 if cos < 0 else 0.0

        if weight == 0.0:
            return g1.clone()

        g1_flat = g1.reshape(-1).float()
        g2_flat = g2.reshape(-1).float()

        g2_norm_sq = (g2_flat @ g2_flat) + 1e-8
        projection_scalar = (g1_flat @ g2_flat) / g2_norm_sq

        resolved_flat = g1_flat - self.config.alpha * weight * projection_scalar * g2_flat
        return resolved_flat.reshape(g1.shape).to(g1.dtype)

    # ------------------------------------------------------------------
    # Multi-task resolution
    # ------------------------------------------------------------------

    def resolve(self, task_grads: list[list[Tensor]]) -> list[list[Tensor]]:
        """Resolve gradient conflicts across tasks.

        Args:
            task_grads: [n_tasks][n_params]  — one gradient tensor per parameter
                        per task.

        Returns:
            resolved_grads: same shape as task_grads with conflicts projected out.
        """
        n_tasks = len(task_grads)
        n_params = len(task_grads[0])

        # Optionally normalise magnitudes before conflict check
        if self.config.normalize_gradients:
            normed: list[list[Tensor]] = []
            for tg in task_grads:
                task_normed: list[Tensor] = []
                for g in tg:
                    norm = g.norm() + 1e-8
                    task_normed.append(g / norm)
                normed.append(task_normed)
            work = normed
        else:
            work = [list(tg) for tg in task_grads]

        # Pairwise projection: for each ordered pair (i, j) project task_i away from task_j
        resolved: list[list[Tensor]] = [list(tg) for tg in work]

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                for p in range(n_params):
                    resolved[i][p] = self.project(resolved[i][p], work[j][p])

        return resolved

    # ------------------------------------------------------------------
    # Per-layer conflict
    # ------------------------------------------------------------------

    def per_layer_conflict(self, task_grads: list[list[Tensor]]) -> list[float]:
        """Compute per-parameter conflict score averaged over all task pairs.

        Returns a list of floats with length == n_params.
        """
        n_tasks = len(task_grads)
        n_params = len(task_grads[0])

        scores: list[float] = []
        for p in range(n_params):
            pair_scores: list[float] = []
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i >= j:
                        continue
                    pair_scores.append(self.conflict_score(task_grads[i][p], task_grads[j][p]))
            scores.append(sum(pair_scores) / len(pair_scores) if pair_scores else 0.0)
        return scores

    # ------------------------------------------------------------------
    # Optimizer integration
    # ------------------------------------------------------------------

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        task_losses: list[Tensor],
    ) -> dict:
        """Compute per-task gradients, resolve conflicts, and apply to params.

        Args:
            optimizer:   a torch optimizer whose param groups hold the model params.
            task_losses: list of scalar loss tensors, one per task.

        Returns:
            {"n_conflicts": int, "mean_conflict_score": float}
        """
        # Collect all parameters managed by this optimizer
        params: list[Tensor] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        # Compute per-task gradients without accumulating into .grad
        task_grads: list[list[Tensor]] = []
        for loss in task_losses:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads: list[Tensor] = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.clone())
                else:
                    grads.append(torch.zeros_like(p))
            task_grads.append(grads)

        # Resolve conflicts
        resolved = self.resolve(task_grads)

        # Count conflicts and accumulate scores
        n_conflicts = 0
        all_scores: list[float] = []
        n_tasks = len(task_grads)
        n_params = len(params)
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i >= j:
                    continue
                for p in range(n_params):
                    score = self.conflict_score(task_grads[i][p], task_grads[j][p])
                    all_scores.append(score)
                    if score < 0:
                        n_conflicts += 1

        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Apply averaged resolved gradients
        optimizer.zero_grad()
        for p_idx, p in enumerate(params):
            avg_grad = torch.stack(
                [resolved[t][p_idx] for t in range(n_tasks)], dim=0
            ).mean(dim=0)
            p.grad = avg_grad.to(p.dtype)

        optimizer.step()

        return {"n_conflicts": n_conflicts, "mean_conflict_score": mean_score}
