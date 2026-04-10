"""DARTS-style differentiable Neural Architecture Search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DARTSConfig:
    """Configuration for DARTS NAS."""
    n_ops: int = 4
    n_cells: int = 4
    d_model: int = 64
    temperature: float = 1.0
    arch_lr: float = 3e-4


# ---------------------------------------------------------------------------
# Mixed Operation
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """Weighted mixture of candidate Linear operations.

    Candidate ops are Linear transformations with hidden sizes:
        d_model//4, d_model//2, d_model, d_model*2 (capped at d_model)
    Each op maps (B, *, d_model) -> (B, *, d_model).
    """

    def __init__(self, d_model: int, n_ops: int) -> None:
        super().__init__()
        hidden_sizes = [
            max(1, d_model // 4),
            max(1, d_model // 2),
            d_model,
            min(d_model, d_model * 2),  # capped at d_model
        ]
        # Use only n_ops entries (up to 4)
        hidden_sizes = hidden_sizes[:n_ops]

        self.ops = nn.ModuleList()
        for h in hidden_sizes:
            if h == d_model:
                # no bottleneck: single linear
                self.ops.append(nn.Linear(d_model, d_model))
            else:
                # bottleneck: down then up
                self.ops.append(nn.Sequential(
                    nn.Linear(d_model, h),
                    nn.ReLU(),
                    nn.Linear(h, d_model),
                ))

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        """Weighted sum of op outputs.

        Args:
            x: input tensor (..., d_model)
            weights: softmax-normalized weights of shape (n_ops,)
        Returns:
            output tensor (..., d_model)
        """
        out = sum(weights[i] * op(x) for i, op in enumerate(self.ops))
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DARTS Cell
# ---------------------------------------------------------------------------

class DARTSCell(nn.Module):
    """Single searchable cell with one MixedOp and learnable arch_params."""

    def __init__(self, d_model: int, n_ops: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.mixed_op = MixedOp(d_model, n_ops)
        self.arch_params = nn.Parameter(torch.zeros(n_ops))

    def forward(self, x: Tensor) -> Tensor:
        weights = F.softmax(self.arch_params / self.temperature, dim=-1)
        return self.mixed_op(x, weights)


# ---------------------------------------------------------------------------
# DARTS Network
# ---------------------------------------------------------------------------

class DARTSNetwork(nn.Module):
    """Stack of DARTSCells forming a searchable network."""

    def __init__(self, config: DARTSConfig) -> None:
        super().__init__()
        self.config = config
        self.cells = nn.ModuleList([
            DARTSCell(config.d_model, config.n_ops, config.temperature)
            for _ in range(config.n_cells)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for cell in self.cells:
            x = cell(x)
        return x

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return list of all architecture parameters from all cells."""
        return [cell.arch_params for cell in self.cells]

    def discretize(self) -> List[int]:
        """Return argmax op index per cell (the 'chosen' operation)."""
        return [int(cell.arch_params.argmax().item()) for cell in self.cells]


# ---------------------------------------------------------------------------
# DARTS Trainer
# ---------------------------------------------------------------------------

class DARTSTrainer:
    """Bi-level optimizer for DARTS: separates weight and arch param updates."""

    def __init__(
        self,
        model_params: List[nn.Parameter],
        arch_params: List[nn.Parameter],
        config: DARTSConfig,
    ) -> None:
        self.weight_optimizer = torch.optim.Adam(model_params, lr=1e-3)
        self.arch_optimizer = torch.optim.Adam(arch_params, lr=config.arch_lr)

    def weight_step(self, loss: Tensor) -> Tensor:
        """Update model (weight) parameters; return scalar loss value."""
        self.weight_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        return loss.detach()

    def arch_step(self, loss: Tensor) -> Tensor:
        """Update architecture parameters; return scalar loss value."""
        self.arch_optimizer.zero_grad()
        loss.backward()
        self.arch_optimizer.step()
        return loss.detach()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_arch_entropy(arch_params: Tensor, temperature: float = 1.0) -> Tensor:
    """Compute entropy of the architecture distribution.

    Args:
        arch_params: raw logits of shape (n_ops,)
        temperature: softmax temperature
    Returns:
        scalar entropy: -sum(p * log(p))
    """
    p = F.softmax(arch_params / temperature, dim=-1)
    # clamp to avoid log(0)
    entropy = -(p * torch.log(p.clamp(min=1e-10))).sum()
    return entropy
