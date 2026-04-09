"""Neural Architecture Search primitives (DARTS/ENAS-style for transformers).

Implements DARTS-style differentiable architecture search where architecture
choices are represented as continuous weights learned via Gumbel-Softmax.

Key idea: Instead of discrete architecture choices, represent the architecture
as a weighted mixture of all candidate operations. Architecture weights (alpha)
are learned alongside model weights via bilevel optimization.

Reference: Liu et al. 2019 "DARTS: Differentiable Architecture Search"
           arXiv:1806.09055
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NASConfig:
    """Configuration for NAS primitives."""

    n_candidates: int = 4       # number of candidate operations per cell
    d_model: int = 64
    temperature: float = 1.0    # Gumbel softmax temperature
    hard: bool = False          # hard vs soft Gumbel-Softmax
    arch_lr: float = 3e-4


@dataclass
class ArchitectureStats:
    """Statistics over the current architecture weight distributions."""

    selected_ops: list[int]   # argmax of arch weights per cell
    entropy: float            # mean entropy of arch weight distributions
    dominance: float          # max arch weight minus mean (how peaked)


# ---------------------------------------------------------------------------
# Gumbel-Softmax
# ---------------------------------------------------------------------------

def gumbel_softmax(logits: Tensor, temperature: float, hard: bool = False) -> Tensor:
    """Gumbel-Softmax relaxation.

    Args:
        logits: shape (n,) unnormalized log-probabilities.
        temperature: temperature for Gumbel-Softmax (>0, lower = sharper).
        hard: if True use straight-through estimator (one-hot forward, soft backward).

    Returns:
        Tensor of shape (n,) summing to 1.
    """
    # Sample Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)

    if hard:
        # Straight-through: one-hot forward, soft backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Straight-through gradient trick
        return y_hard - y_soft.detach() + y_soft

    return y_soft


# ---------------------------------------------------------------------------
# Architecture weight statistics
# ---------------------------------------------------------------------------

def compute_arch_entropy(arch_weights: Tensor) -> float:
    """Compute mean Shannon entropy over rows of arch_weights.

    Args:
        arch_weights: Tensor of shape (K, n_candidates) — rows must sum to 1.

    Returns:
        Mean entropy (float) across the K cells.
    """
    # Clamp to avoid log(0)
    w = arch_weights.clamp(min=1e-20)
    entropy_per_cell = -(w * w.log()).sum(dim=-1)  # (K,)
    return float(entropy_per_cell.mean().item())


def compute_arch_dominance(arch_weights: Tensor) -> float:
    """Mean over cells of (max_weight - mean_weight). Higher = more decisive.

    Args:
        arch_weights: Tensor of shape (K, n_candidates) — rows must sum to 1.

    Returns:
        Dominance score (float).
    """
    max_w = arch_weights.max(dim=-1).values   # (K,)
    mean_w = arch_weights.mean(dim=-1)        # (K,)
    dominance_per_cell = max_w - mean_w       # (K,)
    return float(dominance_per_cell.mean().item())


# ---------------------------------------------------------------------------
# MixedOp
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """Weighted mixture of candidate operations.

    Each candidate op has the same input and output shape.
    The forward pass returns the weighted sum of all ops applied to x,
    where weights are provided externally (from arch logits / Gumbel-Softmax).
    """

    def __init__(self, ops: list[nn.Module]) -> None:
        super().__init__()
        self.ops = nn.ModuleList(ops)

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        """Compute weighted sum of candidate ops.

        Args:
            x: input tensor (any shape).
            weights: shape (n_ops,) — must sum to 1.

        Returns:
            Weighted sum of op(x) outputs, same shape as x.
        """
        out = None
        for i, op in enumerate(self.ops):
            op_out = op(x)
            if out is None:
                out = weights[i] * op_out
            else:
                out = out + weights[i] * op_out
        return out


# ---------------------------------------------------------------------------
# DARTSCell
# ---------------------------------------------------------------------------

class DARTSCell(nn.Module):
    """A single NAS cell with learnable architecture weights.

    Creates n_candidates Linear(d_model, d_model) ops with optional
    ReLU/identity variants and learns architecture weights via Gumbel-Softmax.
    """

    def __init__(self, cfg: NASConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = cfg.n_candidates
        d = cfg.d_model

        # Build candidate ops: alternate between plain linear, linear+ReLU,
        # identity-like (bias-only via zero-weight init), and linear with bias.
        ops: list[nn.Module] = []
        for i in range(n):
            if i % 4 == 0:
                ops.append(nn.Linear(d, d, bias=False))
            elif i % 4 == 1:
                ops.append(nn.Sequential(nn.Linear(d, d, bias=True), nn.ReLU()))
            elif i % 4 == 2:
                ops.append(nn.Linear(d, d, bias=True))
            else:
                # Identity-like: linear initialized near identity
                lin = nn.Linear(d, d, bias=False)
                with torch.no_grad():
                    nn.init.eye_(lin.weight)
                ops.append(lin)

        self.mixed_op = MixedOp(ops)
        # Learnable architecture logits — one per candidate
        self.arch_logits = nn.Parameter(torch.zeros(n))

    def get_weights(self) -> Tensor:
        """Return Gumbel-Softmax weights from arch_logits.

        Returns:
            Tensor of shape (n_candidates,) summing to ~1.
        """
        return gumbel_softmax(self.arch_logits, self.cfg.temperature, self.cfg.hard)

    def forward(self, x: Tensor) -> Tensor:
        """Compute weighted mix of candidate ops applied to x.

        Args:
            x: input tensor of shape (..., d_model).

        Returns:
            Output of same shape as x.
        """
        weights = self.get_weights()
        return self.mixed_op(x, weights)


# ---------------------------------------------------------------------------
# DARTSSearcher
# ---------------------------------------------------------------------------

class DARTSSearcher:
    """Bi-level optimization manager: model weights + architecture weights.

    Maintains separate parameter groups so that architecture parameters
    (arch_logits from cells) can be updated independently of model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        cells: list[DARTSCell],
        cfg: NASConfig,
    ) -> None:
        self.model = model
        self.cells = cells
        self.cfg = cfg

    def arch_parameters(self) -> list[nn.Parameter]:
        """Return list of arch_logits from all cells (one per cell)."""
        return [cell.arch_logits for cell in self.cells]

    def model_parameters(self) -> list[nn.Parameter]:
        """Return model parameters, excluding architecture parameters."""
        arch_param_ids = {id(cell.arch_logits) for cell in self.cells}
        return [p for p in self.model.parameters() if id(p) not in arch_param_ids]

    def get_architecture_stats(self) -> ArchitectureStats:
        """Compute statistics over all cells' architecture weight distributions.

        Returns:
            ArchitectureStats with selected_ops, entropy, and dominance.
        """
        with torch.no_grad():
            # Collect softmax weights (not Gumbel — deterministic for stats)
            weight_rows = []
            selected_ops = []
            for cell in self.cells:
                w = F.softmax(cell.arch_logits, dim=0)
                weight_rows.append(w)
                selected_ops.append(int(w.argmax().item()))

            arch_weights = torch.stack(weight_rows, dim=0)  # (K, n_candidates)
            entropy = compute_arch_entropy(arch_weights)
            dominance = compute_arch_dominance(arch_weights)

        return ArchitectureStats(
            selected_ops=selected_ops,
            entropy=entropy,
            dominance=dominance,
        )

    def discretize(self) -> list[int]:
        """Return argmax of arch_logits for each cell (final discrete architecture).

        Returns:
            List of int indices, one per cell.
        """
        with torch.no_grad():
            return [int(cell.arch_logits.argmax().item()) for cell in self.cells]


# ---------------------------------------------------------------------------
# Random Architecture Search
# ---------------------------------------------------------------------------

def random_architecture_search(
    model: nn.Module,
    cells: list[DARTSCell],
    n_trials: int,
    eval_fn: Callable[[nn.Module], float],
) -> tuple[list[int], float]:
    """Random search baseline over architecture choices.

    For each trial, randomly perturbs arch_logits (uniform random) and
    evaluates the model using eval_fn. Returns the architecture (as list
    of argmax indices) and the best score achieved.

    Args:
        model: the nn.Module to evaluate.
        cells: list of DARTSCells whose arch_logits will be randomized.
        n_trials: number of random architectures to sample.
        eval_fn: callable taking the model and returning a float score
                 (higher is better).

    Returns:
        Tuple of (best_arch: list[int], best_score: float).
    """
    best_arch: list[int] = []
    best_score = float("-inf")

    # Save original logits so we can restore them
    orig_logits = [cell.arch_logits.data.clone() for cell in cells]

    for _ in range(n_trials):
        # Randomize arch logits
        with torch.no_grad():
            for cell in cells:
                cell.arch_logits.data.uniform_(-1.0, 1.0)

        score = eval_fn(model)

        if score > best_score:
            best_score = score
            best_arch = [int(cell.arch_logits.argmax().item()) for cell in cells]

    # Restore original logits
    with torch.no_grad():
        for cell, orig in zip(cells, orig_logits):
            cell.arch_logits.data.copy_(orig)

    return best_arch, best_score
