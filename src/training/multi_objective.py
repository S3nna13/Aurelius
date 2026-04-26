"""Multi-objective optimization: Pareto fronts, hypervolume, and MOO-based loss weighting."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MOOConfig:
    """Configuration for multi-objective optimization."""

    n_objectives: int = 3
    method: str = "linear_scalarization"  # "linear_scalarization" | "chebyshev" | "eps_constrained"
    reference_point: list[float] | None = None  # for hypervolume computation
    eps_tolerance: float = 0.05  # for eps_constrained method


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------


def is_pareto_dominant(a: list[float], b: list[float]) -> bool:
    """Return True if solution *a* dominates solution *b* (minimization).

    *a* dominates *b* when:
      - *a* is no worse than *b* in every objective (a[i] <= b[i] for all i), AND
      - *a* is strictly better in at least one objective (a[i] < b[i] for some i).
    """
    if len(a) != len(b):
        raise ValueError("Solutions must have the same number of objectives.")

    no_worse = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return no_worse and strictly_better


def compute_pareto_front(solutions: list[list[float]]) -> list[int]:
    """Return indices of Pareto-optimal solutions (minimization).

    A solution is Pareto-optimal if no other solution dominates it.

    Args:
        solutions: list of objective vectors, one per solution.

    Returns:
        Sorted list of indices that are on the Pareto front.
    """
    n = len(solutions)
    dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_pareto_dominant(solutions[j], solutions[i]):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Hypervolume indicator
# ---------------------------------------------------------------------------


def compute_hypervolume(pareto_front: list[list[float]], reference_point: list[float]) -> float:
    """Compute the hypervolume indicator for a Pareto front (minimization).

    For 2-D: exact sweep-line algorithm.
    For >2-D: Monte Carlo approximation with 1 000 samples.

    Args:
        pareto_front: list of objective vectors on the Pareto front.
        reference_point: a point dominated by all front members (worst case).

    Returns:
        Hypervolume as a float.
    """
    if not pareto_front:
        return 0.0

    n_obj = len(pareto_front[0])

    if n_obj == 2:
        # Exact 2-D computation: sort by first objective ascending
        sorted_front = sorted(pareto_front, key=lambda p: p[0])
        ref_x, ref_y = reference_point[0], reference_point[1]
        hv = 0.0
        prev_y = ref_y
        for point in sorted_front:
            width = ref_x - point[0]
            height = prev_y - point[1]
            if width > 0 and height > 0:
                hv += width * height
            prev_y = min(prev_y, point[1])
        return hv

    # Monte Carlo approximation for >2 objectives
    n_samples = 1000
    lower = [min(p[k] for p in pareto_front) for k in range(n_obj)]
    upper = reference_point

    # Volume of the bounding box
    box_volume = 1.0
    for lo, hi in zip(lower, upper):
        box_volume *= max(hi - lo, 0.0)

    if box_volume == 0.0:
        return 0.0

    count = 0
    for _ in range(n_samples):
        sample = [random.uniform(lower[k], upper[k]) for k in range(n_obj)]
        # Point is dominated by the front (i.e. inside hypervolume) if any front
        # member dominates it (is no worse on all objectives)
        for fp in pareto_front:
            if all(fp[k] <= sample[k] for k in range(n_obj)):
                count += 1
                break

    return box_volume * count / n_samples


# ---------------------------------------------------------------------------
# Scalarization functions
# ---------------------------------------------------------------------------


def linear_scalarization(objectives: Tensor, weights: Tensor) -> Tensor:
    """Weighted sum of objectives.

    Args:
        objectives: 1-D tensor of objective values, shape (n_objectives,).
        weights: 1-D tensor of non-negative weights, shape (n_objectives,).

    Returns:
        Scalar tensor.
    """
    return (weights * objectives).sum()


def chebyshev_scalarization(
    objectives: Tensor,
    weights: Tensor,
    reference_point: Tensor,
) -> Tensor:
    """Chebyshev scalarization: max_i( weights_i * |objectives_i - reference_i| ).

    Args:
        objectives: 1-D tensor, shape (n_objectives,).
        weights: 1-D tensor, shape (n_objectives,).
        reference_point: 1-D tensor, shape (n_objectives,).

    Returns:
        Scalar tensor.
    """
    return (weights * (objectives - reference_point).abs()).max()


def eps_constrained_loss(
    primary_loss: Tensor,
    constraint_losses: list[Tensor],
    tolerances: list[float],
) -> Tensor:
    """Epsilon-constrained scalarization.

    Optimizes the primary objective subject to epsilon-constraints on the
    remaining objectives.  Violated constraints contribute a squared penalty.

    penalty = sum_i max(0, constraint_losses[i] - tolerances[i])^2

    Args:
        primary_loss: scalar tensor — the main objective to minimise.
        constraint_losses: list of scalar tensors — secondary objectives.
        tolerances: epsilon tolerance per secondary objective.

    Returns:
        Scalar tensor: primary_loss + sum of squared violation penalties.
    """
    penalty = torch.zeros_like(primary_loss)
    for closs, tol in zip(constraint_losses, tolerances):
        violation = torch.clamp(closs - tol, min=0.0)
        penalty = penalty + violation**2
    return primary_loss + penalty


# ---------------------------------------------------------------------------
# MOO Trainer
# ---------------------------------------------------------------------------


class MOOTrainer:
    """Trainer that scalarizes multiple objectives using Pareto-based methods.

    Args:
        model: any nn.Module.
        config: MOOConfig instance.
        optimizer: any torch optimizer wrapping model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MOOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

        # Uniform weights to start
        self._weights = torch.ones(config.n_objectives) / config.n_objectives

        # Reference point for Chebyshev: initialised lazily on first use
        if config.reference_point is not None:
            self._ref = torch.tensor(config.reference_point, dtype=torch.float32)
        else:
            self._ref = torch.zeros(config.n_objectives)

        # History of recent Pareto-front solutions for weight adaptation
        self._recent_losses: list[list[float]] = []

    # ------------------------------------------------------------------
    def train_step(self, objective_fns: list[Callable]) -> dict:
        """One optimisation step over all objectives.

        Args:
            objective_fns: list of callables, each returning a scalar Tensor
                           when called with no arguments (they capture their
                           own data via closures).

        Returns:
            dict with keys:
              "total_loss"      – float, the scalarized combined loss.
              "objectives"      – list[float], individual objective values.
              "pareto_dominated"– bool, True if this solution is dominated
                                  by any solution in recent history.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute individual objectives
        obj_tensors: list[Tensor] = [fn() for fn in objective_fns]
        obj_values: list[float] = [t.item() for t in obj_tensors]
        objectives_t = torch.stack(obj_tensors)

        # Scalarize
        method = self.config.method
        if method == "linear_scalarization":
            total = linear_scalarization(objectives_t, self._weights.to(objectives_t.device))
        elif method == "chebyshev":
            ref = self._ref.to(objectives_t.device)
            total = chebyshev_scalarization(
                objectives_t, self._weights.to(objectives_t.device), ref
            )
        elif method == "eps_constrained":
            primary = obj_tensors[0]
            constraints = obj_tensors[1:]
            tols = [self.config.eps_tolerance] * len(constraints)
            total = eps_constrained_loss(primary, constraints, tols)
        else:
            raise ValueError(f"Unknown MOO method: '{method}'")

        total.backward()
        self.optimizer.step()

        # Record for Pareto tracking
        self._recent_losses.append(obj_values)

        # Check if this solution is dominated
        pareto_dominated = False
        if len(self._recent_losses) > 1:
            front_indices = compute_pareto_front(self._recent_losses)
            last_idx = len(self._recent_losses) - 1
            pareto_dominated = last_idx not in front_indices

        return {
            "total_loss": total.item(),
            "objectives": obj_values,
            "pareto_dominated": pareto_dominated,
        }

    # ------------------------------------------------------------------
    def update_weights_by_loss(self, recent_losses: list[list[float]]) -> None:
        """Update scalarization weights based on recent Pareto front.

        Strategy: weight each objective inversely proportional to the mean
        value of that objective on the Pareto front.  This encourages the
        optimiser to focus on objectives that are currently large.

        Args:
            recent_losses: list of objective vectors (one per recent step).
        """
        if not recent_losses:
            return

        pareto_indices = compute_pareto_front(recent_losses)
        if not pareto_indices:
            return

        pareto_points = [recent_losses[i] for i in pareto_indices]
        n_obj = len(pareto_points[0])

        # Mean value per objective across the Pareto front
        means = [sum(p[k] for p in pareto_points) / len(pareto_points) for k in range(n_obj)]

        # Inverse-proportional weights; guard against zero means
        inv = [1.0 / (m + 1e-8) for m in means]
        total_inv = sum(inv)
        new_weights = [v / total_inv for v in inv]

        self._weights = torch.tensor(new_weights, dtype=torch.float32)
        logger.debug("MOOTrainer: updated weights to %s", new_weights)
