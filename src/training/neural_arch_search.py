"""Differentiable neural architecture search (DARTS-style) for transformer config selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NASConfig:
    """Configuration for neural architecture search."""

    search_space: dict = field(
        default_factory=lambda: {
            "d_ff_mult": [2, 4, 8],
            "n_heads": [2, 4, 8],
            "activation": ["swiglu", "gelu", "relu"],
        }
    )
    temperature: float = 1.0
    anneal_rate: float = 0.01
    min_temperature: float = 0.1


class ArchitectureWeights(nn.Module):
    """Learnable architecture parameters with one softmax distribution per search dimension."""

    def __init__(self, search_space: dict) -> None:
        super().__init__()
        self.search_space = search_space
        self.arch_params = nn.ParameterDict()
        for key, options in search_space.items():
            self.arch_params[key] = nn.Parameter(torch.zeros(len(options)))

    def get_probs(self) -> dict[str, Tensor]:
        """Return softmax probabilities for each architecture dimension."""
        return {
            key: F.softmax(param, dim=0)
            for key, param in self.arch_params.items()
        }

    def get_best(self) -> dict[str, Any]:
        """Return argmax choice for each dimension."""
        result = {}
        for key, param in self.arch_params.items():
            idx = param.argmax().item()
            result[key] = self.search_space[key][idx]
        return result

    def entropy(self) -> Tensor:
        """Mean entropy across all dimensions (scalar)."""
        entropies = []
        for param in self.arch_params.values():
            probs = F.softmax(param, dim=0)
            log_probs = F.log_softmax(param, dim=0)
            entropies.append(-(probs * log_probs).sum())
        return torch.stack(entropies).mean()


def gumbel_softmax_sample(
    logits: Tensor, temperature: float, hard: bool = False
) -> Tensor:
    """Standard Gumbel-softmax sampling.

    Adds Gumbel noise, divides by temperature, applies softmax.
    If hard: straight-through estimator (argmax forward, soft backward).
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)

    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class MixedOperation(nn.Module):
    """Weighted combination of candidate operations using Gumbel-softmax."""

    def __init__(self, operations: list[nn.Module], arch_logits: nn.Parameter) -> None:
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.arch_logits = arch_logits

    def forward(self, x: Tensor, temperature: float) -> Tensor:
        """Forward pass with Gumbel-softmax weighted combination."""
        weights = gumbel_softmax_sample(self.arch_logits, temperature)
        output = torch.zeros_like(x)
        for i, op in enumerate(self.operations):
            output = output + weights[i] * op(x)
        return output


class NASFFNCandidate(nn.Module):
    """A single FFN candidate with configurable width and activation."""

    def __init__(self, d_model: int, d_ff: int, activation: str) -> None:
        super().__init__()
        self.activation_name = activation

        if activation == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_name == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w_gate(x))
        return self.w2(self.act(self.w1(x)))


class NASSearcher:
    """Orchestrate differentiable neural architecture search."""

    def __init__(self, config: NASConfig, d_model: int) -> None:
        self.config = config
        self.d_model = d_model
        self.temperature = config.temperature

        self.arch_weights = ArchitectureWeights(config.search_space)
        self.candidates = self.build_candidates()

        # Single set of arch logits for the mixed operation over all candidates
        self.arch_logits = nn.Parameter(torch.zeros(len(self.candidates)))
        self.mixed_op = MixedOperation(
            list(self.candidates), self.arch_logits
        )

    def build_candidates(self) -> nn.ModuleList:
        """Create all FFN candidates from search space combinations."""
        candidates = []
        d_ff_mults = self.config.search_space.get("d_ff_mult", [4])
        activations = self.config.search_space.get("activation", ["gelu"])

        for mult in d_ff_mults:
            for act in activations:
                d_ff = self.d_model * mult
                candidates.append(NASFFNCandidate(self.d_model, d_ff, act))

        return nn.ModuleList(candidates)

    def search_step(self, x: Tensor) -> tuple[Tensor, dict]:
        """Forward through MixedOperation and return output with metadata."""
        output = self.mixed_op(x, self.temperature)
        entropy_val = self.arch_weights.entropy()
        best_config = self.arch_weights.get_best()

        info = {
            "entropy": entropy_val.item(),
            "temperature": self.temperature,
            "best_config": best_config,
        }
        return output, info

    def anneal_temperature(self) -> None:
        """Decrease temperature by annealing rate."""
        self.temperature = max(
            self.temperature * (1 - self.config.anneal_rate),
            self.config.min_temperature,
        )

    def get_best_architecture(self) -> dict:
        """Return current best architecture choices."""
        return self.arch_weights.get_best()
