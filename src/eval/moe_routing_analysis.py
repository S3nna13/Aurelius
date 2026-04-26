"""Mixture-of-Experts routing analysis tools.

Diagnoses routing behavior in MoE layers: load balance, entropy,
specialization, and collapse. Distinct from moe_analysis.py which
focuses on capacity and loss; this module examines routing patterns.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# RoutingStats
# ---------------------------------------------------------------------------


class RoutingStats:
    """Accumulates per-expert token assignment statistics over many batches."""

    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self.total_tokens: int = 0
        self.expert_counts: Tensor = torch.zeros(n_experts, dtype=torch.long)

    def update(self, routing_weights: Tensor) -> None:
        """Accumulate counts from a batch of routing weights.

        Args:
            routing_weights: ``(B*T, n_experts)`` softmax router outputs.
                Hard assignment is determined by argmax.
        """
        assignments = routing_weights.argmax(dim=-1)  # (N,)
        n = assignments.shape[0]
        self.total_tokens += n
        counts = torch.bincount(assignments, minlength=self.n_experts)
        self.expert_counts = self.expert_counts + counts.to(self.expert_counts.device)

    def load_balance_score(self) -> float:
        """Coefficient of variation of expert token counts.

        Lower is better (0.0 = perfectly balanced).
        """
        counts = self.expert_counts.float()
        mean = counts.mean()
        std = counts.std(unbiased=False)
        return (std / (mean + 1e-8)).item()

    def router_collapse(self) -> bool:
        """True if any single expert handles > 90 % of all tokens."""
        if self.total_tokens == 0:
            return False
        utilization = self.expert_counts.float() / (self.total_tokens + 1e-8)
        return bool((utilization > 0.9).any().item())

    def expert_utilization(self) -> Tensor:
        """Fraction of tokens routed to each expert.

        Returns:
            Tensor of shape ``(n_experts,)``.
        """
        return self.expert_counts.float() / (self.total_tokens + 1e-8)

    def reset(self) -> None:
        """Zero out accumulated counts."""
        self.total_tokens = 0
        self.expert_counts = torch.zeros(self.n_experts, dtype=torch.long)


# ---------------------------------------------------------------------------
# RoutingEntropyAnalyzer
# ---------------------------------------------------------------------------


class RoutingEntropyAnalyzer:
    """Stateless entropy analysis of per-token routing distributions."""

    def __init__(self) -> None:
        pass

    def token_entropy(self, routing_weights: Tensor) -> Tensor:
        """Per-token entropy of routing distributions.

        Args:
            routing_weights: ``(N, n_experts)`` softmax weights.

        Returns:
            ``(N,)`` tensor of per-token entropies.
        """
        p = routing_weights
        return -(p * torch.log(p + 1e-10)).sum(dim=-1)

    def mean_routing_entropy(self, routing_weights: Tensor) -> float:
        """Mean entropy across all tokens."""
        return self.token_entropy(routing_weights).mean().item()

    def max_entropy(self, n_experts: int) -> float:
        """Theoretical maximum entropy for *n_experts* experts: log(n_experts)."""
        return math.log(n_experts)

    def entropy_efficiency(self, routing_weights: Tensor) -> float:
        """Ratio of mean entropy to maximum entropy.

        1.0 means perfectly uniform routing across experts.
        """
        n_experts = routing_weights.shape[-1]
        return self.mean_routing_entropy(routing_weights) / (self.max_entropy(n_experts) + 1e-10)


# ---------------------------------------------------------------------------
# ExpertSpecializationAnalyzer
# ---------------------------------------------------------------------------


class ExpertSpecializationAnalyzer:
    """Tracks co-occurrence of token types and expert assignments."""

    def __init__(self, n_experts: int, n_token_types: int) -> None:
        self.n_experts = n_experts
        self.n_token_types = n_token_types
        self.co_occurrence: Tensor = torch.zeros(n_token_types, n_experts, dtype=torch.long)

    def update(self, routing_weights: Tensor, token_type_ids: Tensor) -> None:
        """Accumulate co-occurrence counts.

        Args:
            routing_weights: ``(N, n_experts)`` softmax weights.
            token_type_ids: ``(N,)`` long tensor of token type categories.
        """
        assignments = routing_weights.argmax(dim=-1)  # (N,)
        for t in range(self.n_token_types):
            mask = token_type_ids == t
            if mask.any():
                expert_subset = assignments[mask]
                counts = torch.bincount(expert_subset, minlength=self.n_experts)
                self.co_occurrence[t] += counts.to(self.co_occurrence.device)

    def specialization_score(self) -> float:
        """Mean max fraction of tokens per token type assigned to a single expert.

        Returns a value in [0, 1]; higher means stronger specialization.
        """
        row_normalized = self.co_occurrence.float() / (
            self.co_occurrence.sum(dim=1, keepdim=True).float() + 1e-8
        )
        return row_normalized.max(dim=1).values.mean().item()

    def reset(self) -> None:
        """Zero out co-occurrence counts."""
        self.co_occurrence = torch.zeros(self.n_token_types, self.n_experts, dtype=torch.long)


# ---------------------------------------------------------------------------
# RoutingDiagnostics
# ---------------------------------------------------------------------------


class RoutingDiagnostics:
    """Combined routing diagnostics — stateless per call to ``analyze``."""

    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self._stats = RoutingStats(n_experts)
        self._entropy = RoutingEntropyAnalyzer()

    def analyze(self, routing_weights: Tensor) -> dict[str, float]:
        """Compute all routing metrics for one batch of routing weights.

        Args:
            routing_weights: ``(N, n_experts)`` softmax outputs.

        Returns:
            Dictionary with keys:
            ``load_balance_score``, ``router_collapse``, ``mean_entropy``,
            ``entropy_efficiency``, ``min_expert_fraction``,
            ``max_expert_fraction``.
        """
        self._stats.reset()
        self._stats.update(routing_weights)

        utilization = self._stats.expert_utilization()

        result: dict[str, float] = {
            "load_balance_score": self._stats.load_balance_score(),
            "router_collapse": self._stats.router_collapse(),
            "mean_entropy": self._entropy.mean_routing_entropy(routing_weights),
            "entropy_efficiency": self._entropy.entropy_efficiency(routing_weights),
            "min_expert_fraction": utilization.min().item(),
            "max_expert_fraction": utilization.max().item(),
        }

        self._stats.reset()
        return result
