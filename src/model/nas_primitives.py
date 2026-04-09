"""Neural Architecture Search primitives: mixed operations, weight sharing, and DARTS-style search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NASConfig:
    """Configuration for NAS search space and training."""
    n_ops: int = 4                  # number of candidate operations
    d_model: int = 64
    temperature: float = 1.0        # Gumbel-softmax temperature
    straight_through: bool = True   # use straight-through gradient estimator


# ---------------------------------------------------------------------------
# Base operations
# ---------------------------------------------------------------------------

class NASOperation(nn.Module):
    """Abstract base class for NAS candidate operations."""
    name: str = "base"

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class IdentityOp(NASOperation):
    """Pass-through operation: output equals input."""
    name: str = "identity"

    def forward(self, x: Tensor) -> Tensor:
        return x


class ZeroOp(NASOperation):
    """Zero operation: output is all zeros with same shape as input."""
    name: str = "zero"

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class LinearOp(NASOperation):
    """Linear projection operation."""
    name: str = "linear"

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ConvOp(NASOperation):
    """1-D convolution operation over the sequence dimension."""
    name: str = "conv"

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, D, T) -> conv -> (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Mixed operation
# ---------------------------------------------------------------------------

class MixedOperation(nn.Module):
    """Combines multiple NAS operations with learnable architecture weights."""

    def __init__(self, ops: list[NASOperation], config: NASConfig) -> None:
        super().__init__()
        self.config = config
        self.ops = nn.ModuleList(ops)
        # Architecture weights — uniform initialisation
        self.arch_weights = nn.Parameter(torch.ones(len(ops)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalized_weights(self) -> Tensor:
        return F.softmax(self.arch_weights / self.config.temperature, dim=0)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Soft weighted sum of all operation outputs."""
        weights = self._normalized_weights()
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out  # (B, T, D)

    def gumbel_forward(self, x: Tensor) -> Tensor:
        """Discrete (approximately one-hot) weighted sum via Gumbel-softmax."""
        weights = F.gumbel_softmax(
            self.arch_weights,
            tau=self.config.temperature,
            hard=self.config.straight_through,
        )
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out  # (B, T, D)

    # ------------------------------------------------------------------
    # Architecture introspection
    # ------------------------------------------------------------------

    def get_best_op(self) -> NASOperation:
        """Return the operation with the highest architecture weight."""
        idx = int(self.arch_weights.argmax().item())
        return self.ops[idx]  # type: ignore[return-value]

    def architecture_entropy(self) -> float:
        """Shannon entropy of the softmax-normalised architecture weights."""
        probs = self._normalized_weights()
        entropy = -(probs * probs.log()).sum()
        return float(entropy.item())


# ---------------------------------------------------------------------------
# DARTS Cell
# ---------------------------------------------------------------------------

def _default_config() -> NASConfig:
    return NASConfig()


class DARTSCell(nn.Module):
    """DARTS-style cell with mixed operations between every pair of nodes."""

    def __init__(
        self,
        n_nodes: int = 3,
        d_model: int = 64,
        config: NASConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.config = config if config is not None else NASConfig(d_model=d_model)

        # ops[i][j]: MixedOperation from node j -> node i
        # Node indices: 0 = cell input; 1..n_nodes = intermediate nodes
        # Node i (1-indexed) receives input from nodes 0..(i-1)
        self.ops: nn.ModuleList = nn.ModuleList()
        for i in range(1, n_nodes + 1):
            node_ops = nn.ModuleList()
            for _j in range(i):  # predecessors: 0 .. i-1
                candidate_ops = self._build_ops(d_model)
                mixed = MixedOperation(candidate_ops, self.config)
                node_ops.append(mixed)
            self.ops.append(node_ops)

        # Project concatenated node outputs back to d_model
        self.proj = nn.Linear(n_nodes * d_model, d_model)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_ops(d_model: int) -> list[NASOperation]:
        return [
            IdentityOp(),
            ZeroOp(),
            LinearOp(d_model),
            ConvOp(d_model),
        ]

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D) — cell input (node 0)
        Returns:
            (B, T, D)
        """
        nodes: list[Tensor] = [x]  # node 0 is the input

        for i in range(self.n_nodes):
            # node_ops[i] contains MixedOps from predecessors 0..i
            node_ops: nn.ModuleList = self.ops[i]  # type: ignore[assignment]
            h = sum(op(nodes[j]) for j, op in enumerate(node_ops))
            nodes.append(h)  # type: ignore[arg-type]

        # Concatenate intermediate node outputs (exclude input node 0)
        intermediate = torch.cat(nodes[1:], dim=-1)  # (B, T, n_nodes*D)
        return self.proj(intermediate)  # (B, T, D)

    # ------------------------------------------------------------------

    def get_architecture_weights(self) -> dict[str, Tensor]:
        """Return all arch_weights tensors keyed by edge name."""
        weights: dict[str, Tensor] = {}
        for i in range(self.n_nodes):
            node_ops: nn.ModuleList = self.ops[i]  # type: ignore[assignment]
            for j, op in enumerate(node_ops):
                edge = f"node{j}->node{i + 1}"
                weights[edge] = op.arch_weights  # type: ignore[union-attr]
        return weights


# ---------------------------------------------------------------------------
# Discretize architecture
# ---------------------------------------------------------------------------

def discretize_architecture(cell: DARTSCell, top_k: int = 2) -> dict[str, str]:
    """Select the top-k operation per node edge by architecture weight.

    Returns a dict mapping edge names to the name of the selected operation.
    When top_k == 1 the single best op is returned per edge.
    """
    result: dict[str, str] = {}
    arch_weights = cell.get_architecture_weights()
    for edge, weights in arch_weights.items():
        # Find op indices sorted by weight (descending)
        sorted_indices = weights.argsort(descending=True)
        i = int(edge.split("->")[1].replace("node", "")) - 1  # 0-indexed destination node
        j = int(edge.split("->")[0].replace("node", ""))       # source node index
        mixed_op: MixedOperation = cell.ops[i][j]  # type: ignore[assignment,index]
        for k in range(min(top_k, len(sorted_indices))):
            idx = int(sorted_indices[k].item())
            op: NASOperation = mixed_op.ops[idx]  # type: ignore[assignment]
            key = edge if top_k == 1 else f"{edge}_top{k + 1}"
            result[key] = op.name
    return result


# ---------------------------------------------------------------------------
# NAS Architecture Optimizer
# ---------------------------------------------------------------------------

class NASArchitectureOptimizer:
    """Separate optimizers for architecture weights and model weights (DARTS style)."""

    def __init__(
        self,
        cell: DARTSCell,
        arch_lr: float = 3e-4,
        weight_lr: float = 3e-4,
    ) -> None:
        self.cell = cell

        arch_ids = {id(p) for p in self.arch_params()}
        weight_ps = [p for p in cell.parameters() if id(p) not in arch_ids]

        self.arch_optimizer = torch.optim.Adam(self.arch_params(), lr=arch_lr)
        self.weight_optimizer = torch.optim.Adam(weight_ps, lr=weight_lr)

    # ------------------------------------------------------------------

    def arch_params(self) -> list[nn.Parameter]:
        """All architecture weight parameters."""
        params: list[nn.Parameter] = []
        for i in range(self.cell.n_nodes):
            node_ops: nn.ModuleList = self.cell.ops[i]  # type: ignore[assignment]
            for op in node_ops:
                params.append(op.arch_weights)  # type: ignore[union-attr]
        return params

    def weight_params(self) -> list[nn.Parameter]:
        """All non-architecture parameters."""
        arch_ids = {id(p) for p in self.arch_params()}
        return [p for p in self.cell.parameters() if id(p) not in arch_ids]

    # ------------------------------------------------------------------

    def step_weights(self, loss: Tensor) -> None:
        """Backward + weight optimizer step."""
        self.weight_optimizer.zero_grad()
        loss.backward()
        self.weight_optimizer.step()

    def step_arch(self, loss: Tensor) -> None:
        """Backward + architecture optimizer step."""
        self.arch_optimizer.zero_grad()
        loss.backward()
        self.arch_optimizer.step()
