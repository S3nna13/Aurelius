"""DARTS-style differentiable Neural Architecture Search with Gumbel-softmax."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NASConfig:
    """Configuration for DARTS-style NAS search."""

    n_ops: int = 4
    d_model: int = 64
    temperature: float = 1.0
    hard: bool = False


# ---------------------------------------------------------------------------
# Gumbel-softmax helper
# ---------------------------------------------------------------------------


def gumbel_softmax_weights(logits: Tensor, temperature: float, hard: bool = False) -> Tensor:
    """Compute Gumbel-softmax weights.

    Adds Gumbel noise to logits, then applies softmax.  When ``hard=True`` the
    forward pass returns a one-hot vector while the backward pass flows through
    the soft weights (straight-through estimator).

    Args:
        logits: raw architecture logits of shape (n_ops,)
        temperature: softmax temperature τ > 0
        hard: if True, use straight-through estimator

    Returns:
        Weights tensor of same shape as *logits*, summing to 1.
    """
    # Add Gumbel noise: -log(-log(U))  where U ~ Uniform(0,1)
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0,1)
    y = (logits + gumbels) / temperature
    soft = F.softmax(y, dim=-1)

    if not hard:
        return soft

    # Straight-through: one-hot forward, soft backward
    index = soft.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(soft).scatter_(-1, index, 1.0)
    # straight-through trick: detach the difference so gradients flow through soft
    return one_hot - soft.detach() + soft


# ---------------------------------------------------------------------------
# Mixed Operation
# ---------------------------------------------------------------------------


class MixedOp(nn.Module):
    """Weighted mixture of candidate operations with learnable architecture weights."""

    def __init__(self, ops: nn.ModuleList, config: NASConfig) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        n = len(ops)
        self.arch_weights = nn.Parameter(torch.zeros(n))

    def forward(self, x: Tensor) -> Tensor:
        """Gumbel-softmax weighted sum of all candidate operation outputs."""
        weights = gumbel_softmax_weights(
            self.arch_weights,
            temperature=self.config.temperature,
            hard=self.config.hard,
        )
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DARTS Cell
# ---------------------------------------------------------------------------


def _build_candidate_ops(d_model: int) -> nn.ModuleList:
    """Build the list of candidate operations for a cell."""
    relu_linear = nn.Sequential(nn.ReLU(), nn.Linear(d_model, d_model))
    norm_linear = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
    return nn.ModuleList(
        [
            nn.Identity(),
            nn.Linear(d_model, d_model),
            relu_linear,
            norm_linear,
        ]
    )


class DARTSCell(nn.Module):
    """A single searchable cell with one MixedOp over candidate operations."""

    def __init__(self, d_model: int, n_ops: int, config: NASConfig) -> None:
        super().__init__()
        self.config = config
        ops = _build_candidate_ops(d_model)
        # Truncate or use only n_ops operations
        truncated = nn.ModuleList(list(ops)[:n_ops])
        self.mixed_op = MixedOp(truncated, config)

    def forward(self, x: Tensor) -> Tensor:
        return self.mixed_op(x)

    def get_arch_params(self) -> list[nn.Parameter]:
        return [self.mixed_op.arch_weights]


# ---------------------------------------------------------------------------
# DARTS Searcher
# ---------------------------------------------------------------------------


class DARTSSearcher:
    """Manages bi-level optimization for a collection of DARTS cells."""

    def __init__(self, cells: nn.ModuleList, config: NASConfig) -> None:
        self.cells = cells
        self.config = config

    def arch_params(self) -> list[nn.Parameter]:
        """Collect all architecture parameters from every cell."""
        params: list[nn.Parameter] = []
        for cell in self.cells:
            if hasattr(cell, "get_arch_params"):
                params.extend(cell.get_arch_params())
        return params

    def model_params(self) -> list[nn.Parameter]:
        """Collect all non-architecture parameters from every cell."""
        arch_ids = {id(p) for p in self.arch_params()}
        params: list[nn.Parameter] = []
        for cell in self.cells:
            for p in cell.parameters():
                if id(p) not in arch_ids:
                    params.append(p)
        return params

    def discretize(self) -> list[int]:
        """Return the index of the dominant operation per cell (argmax of arch_weights)."""
        result: list[int] = []
        for cell in self.cells:
            if hasattr(cell, "get_arch_params"):
                arch_param = cell.get_arch_params()[0]
                result.append(int(arch_param.argmax().item()))
            else:
                result.append(0)
        return result
