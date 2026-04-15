"""Multi-objective reward modeling with Pareto-based and scalarization approaches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadRewardModel(nn.Module):
    """Single backbone hidden state -> N scalar reward heads.

    The backbone is external; this module takes pre-computed hidden states
    of shape (B, hidden_dim) and projects each through an independent linear
    head to produce per-objective rewards.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_objectives: int,
        objective_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_objectives = n_objectives
        self.objective_names = objective_names or [f"obj_{i}" for i in range(n_objectives)]

        # Independent linear head per objective
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_objectives)]
        )
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Returns rewards: (B, n_objectives).

        Args:
            hidden_states: (B, hidden_dim)
        """
        rewards = torch.cat(
            [head(hidden_states) for head in self.heads], dim=-1
        )  # (B, n_objectives)
        return rewards

    def get_objective_reward(self, hidden_states: Tensor, objective_idx: int) -> Tensor:
        """Returns scalar reward for one objective: (B,).

        Args:
            hidden_states: (B, hidden_dim)
            objective_idx: index into [0, n_objectives)
        """
        return self.heads[objective_idx](hidden_states).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Scalarization strategies
# ---------------------------------------------------------------------------

def linear_scalarize(rewards: Tensor, weights: Tensor) -> Tensor:
    """Weighted sum scalarization.

    Args:
        rewards: (B, K)
        weights: (K,)

    Returns:
        (B,)
    """
    weights = weights.to(rewards.device, rewards.dtype)
    return (rewards * weights.unsqueeze(0)).sum(dim=-1)


def chebyshev_scalarize(
    rewards: Tensor,
    weights: Tensor,
    reference_point: Optional[Tensor] = None,
) -> Tensor:
    """Chebyshev (minimax) scalarization. Encourages Pareto-diverse solutions.

    Scalarized value = -max_k( weights_k * |rewards_k - ref_k| )
    We negate so that higher is better (consistent with reward maximization).

    Args:
        rewards: (B, K)
        weights: (K,)
        reference_point: (K,) utopian/reference point; defaults to zeros.

    Returns:
        (B,)
    """
    weights = weights.to(rewards.device, rewards.dtype)
    if reference_point is None:
        reference_point = torch.zeros(rewards.shape[-1], device=rewards.device, dtype=rewards.dtype)
    else:
        reference_point = reference_point.to(rewards.device, rewards.dtype)

    # Weighted absolute deviation from reference (B, K)
    dev = weights.unsqueeze(0) * (rewards - reference_point.unsqueeze(0)).abs()
    # Minimax: take the maximum deviation per sample, negate so higher = better
    return -dev.max(dim=-1).values  # (B,)


def hypervolume_scalarize(rewards: Tensor, weights: Tensor) -> Tensor:
    """Hypervolume-inspired scalarization approximation.

    Uses the product of weighted rewards as a proxy for dominated hypervolume.
    Rewards are shifted to be non-negative before taking the product.

    Args:
        rewards: (B, K)
        weights: (K,)

    Returns:
        (B,)
    """
    weights = weights.to(rewards.device, rewards.dtype)
    # Shift to non-negative
    shifted = rewards - rewards.min(dim=0).values.unsqueeze(0) + 1e-6
    weighted = shifted ** weights.unsqueeze(0)
    return weighted.prod(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Pareto dominance utilities
# ---------------------------------------------------------------------------

def is_pareto_dominated(solution: Tensor, population: Tensor) -> bool:
    """Returns True if any row in population dominates solution on ALL objectives.

    Dominance: dominator >= solution on every objective AND strictly > on at least one.

    Args:
        solution: (K,)
        population: (N, K)

    Returns:
        bool
    """
    if population.shape[0] == 0:
        return False

    # dominator must be >= on all objectives
    ge_all = (population >= solution.unsqueeze(0)).all(dim=-1)  # (N,)
    # and strictly > on at least one
    gt_any = (population > solution.unsqueeze(0)).any(dim=-1)   # (N,)

    return bool((ge_all & gt_any).any().item())


def pareto_front(solutions: Tensor) -> Tensor:
    """Returns boolean mask (N,) of non-dominated solutions.

    Args:
        solutions: (N, K)

    Returns:
        (N,) bool tensor; True = non-dominated (on Pareto front)
    """
    n = solutions.shape[0]
    is_dominated = torch.zeros(n, dtype=torch.bool, device=solutions.device)

    for i in range(n):
        if is_dominated[i]:
            continue
        others = solutions  # (N, K)
        # Check if solution i is dominated by any other
        # We skip self-comparison by checking only strict improvement
        for j in range(n):
            if i == j:
                continue
            if is_dominated[i]:
                break
            # j dominates i?
            if (solutions[j] >= solutions[i]).all() and (solutions[j] > solutions[i]).any():
                is_dominated[i] = True

    return ~is_dominated


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveConfig:
    name: str
    weight: float = 1.0
    minimize: bool = False  # True for loss-type objectives (lower is better)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultiObjectiveRMTrainer:
    """Trains a MultiHeadRewardModel with per-objective Bradley-Terry losses."""

    def __init__(
        self,
        model: MultiHeadRewardModel,
        objectives: List[ObjectiveConfig],
        scalarization: str = "linear",  # "linear" | "chebyshev"
        lr: float = 1e-4,
    ) -> None:
        if len(objectives) != model.n_objectives:
            raise ValueError(
                f"len(objectives)={len(objectives)} must equal model.n_objectives={model.n_objectives}"
            )
        self.model = model
        self.objectives = objectives
        self.scalarization = scalarization
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Build weight tensor from objective configs
        self._weights = torch.tensor(
            [obj.weight for obj in objectives], dtype=torch.float32
        )

    def _scalarize(self, losses: Tensor) -> Tensor:
        """losses: (n_objectives,) -> scalar."""
        if self.scalarization == "chebyshev":
            # chebyshev over losses (B=1 view)
            return chebyshev_scalarize(
                losses.unsqueeze(0), self._weights.to(losses.device)
            ).squeeze(0).neg()  # negate back since chebyshev negates
        else:
            return linear_scalarize(
                losses.unsqueeze(0), self._weights.to(losses.device)
            ).squeeze(0)

    def compute_loss(
        self,
        chosen_hidden: Tensor,    # (B, hidden_dim)
        rejected_hidden: Tensor,  # (B, hidden_dim)
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Bradley-Terry loss per objective, then scalarized.

        Returns (total_loss, per_objective_metrics).
        """
        chosen_rewards = self.model(chosen_hidden)    # (B, n_obj)
        rejected_rewards = self.model(rejected_hidden)  # (B, n_obj)

        per_obj_losses: List[Tensor] = []
        metrics: Dict[str, float] = {}

        for idx, obj in enumerate(self.objectives):
            r_chosen = chosen_rewards[:, idx]    # (B,)
            r_rejected = rejected_rewards[:, idx]  # (B,)

            if obj.minimize:
                # For minimize objectives: preferred = lower score
                loss = -F.logsigmoid(r_rejected - r_chosen).mean()
            else:
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            per_obj_losses.append(loss)
            metrics[obj.name] = loss.item()

        losses_tensor = torch.stack(per_obj_losses)  # (n_objectives,)
        total_loss = self._scalarize(losses_tensor)

        return total_loss, metrics

    def detect_gradient_conflict(
        self,
        chosen_hidden: Tensor,
        rejected_hidden: Tensor,
    ) -> Tensor:
        """Returns (n_objectives, n_objectives) cosine similarity matrix of per-objective gradients."""
        n_obj = self.model.n_objectives
        grads: List[Tensor] = []

        chosen_rewards = self.model(chosen_hidden)    # (B, n_obj)
        rejected_rewards = self.model(rejected_hidden)  # (B, n_obj)

        for idx, obj in enumerate(self.objectives):
            r_chosen = chosen_rewards[:, idx]
            r_rejected = rejected_rewards[:, idx]

            if obj.minimize:
                loss = -F.logsigmoid(r_rejected - r_chosen).mean()
            else:
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            # Compute gradients w.r.t. all model parameters
            grad_list = torch.autograd.grad(
                loss,
                list(self.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            # Flatten and concatenate, replacing None with zeros
            flat_grads = []
            for g, p in zip(grad_list, self.model.parameters()):
                if g is None:
                    flat_grads.append(torch.zeros_like(p).view(-1))
                else:
                    flat_grads.append(g.view(-1))
            grads.append(torch.cat(flat_grads))

        # Build cosine similarity matrix
        sim_matrix = torch.zeros(n_obj, n_obj)
        for i in range(n_obj):
            for j in range(n_obj):
                gi = grads[i]
                gj = grads[j]
                cos = F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0)).item()
                sim_matrix[i, j] = cos

        return sim_matrix


# ---------------------------------------------------------------------------
# Pareto improvement reward
# ---------------------------------------------------------------------------

def compute_pareto_reward(
    rewards: Tensor,            # (B, K)
    reference_rewards: Tensor,  # (B, K)
) -> Tensor:
    """Compute Pareto improvement reward per sample.

    Returns (B,) integer tensor:
      +1 if rewards[b] Pareto-dominates reference_rewards[b]
      -1 if dominated by reference_rewards[b]
       0 otherwise
    """
    B = rewards.shape[0]
    result = torch.zeros(B, dtype=torch.long, device=rewards.device)

    for b in range(B):
        r = rewards[b]      # (K,)
        ref = reference_rewards[b]  # (K,)

        # Check if r dominates ref
        if (r >= ref).all() and (r > ref).any():
            result[b] = 1
        # Check if ref dominates r
        elif (ref >= r).all() and (ref > r).any():
            result[b] = -1
        # else 0

    return result
