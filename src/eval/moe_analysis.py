"""MoE routing analysis — profile expert utilization patterns, load balance,
and specialization in Mixture-of-Experts models.

Works on any model; simulates expert assignment for dense models via k-means-style
token clustering using a fixed random projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MoEAnalysisConfig:
    """Configuration for MoE analysis."""

    n_experts: int = 8
    top_k: int = 2
    n_samples: int = 50
    track_layers: list[int] = field(default_factory=list)  # empty = all layers


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExpertStats:
    """Per-expert statistics for one layer."""

    expert_id: int
    utilization: float  # fraction of tokens assigned to this expert
    mean_activation: float  # mean activation magnitude at this expert
    specialization_score: float  # how concentrated assignments are to certain token types


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def simulate_expert_routing(
    hidden_states: torch.Tensor,
    n_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate MoE routing for a dense model using a fixed random projection.

    Projects hidden_states (B, T, D) to expert logits via a deterministic random
    linear projection (D -> n_experts), applies softmax, then selects the top_k
    experts per token.

    Args:
        hidden_states: (B, T, D) hidden state tensor.
        n_experts: number of simulated experts.
        top_k: number of experts selected per token.

    Returns:
        expert_indices: (B, T, top_k) — indices of selected experts.
        expert_weights: (B, T, top_k) — softmax weights for selected experts.
    """
    B, T, D = hidden_states.shape

    # Deterministic fixed projection
    gen = torch.Generator()
    gen.manual_seed(42)
    W = torch.randn(
        D, n_experts, generator=gen, dtype=hidden_states.dtype, device=hidden_states.device
    )

    # (B, T, D) @ (D, n_experts) -> (B, T, n_experts)
    logits = hidden_states @ W
    probs = F.softmax(logits, dim=-1)  # (B, T, n_experts)

    # Top-k selection
    expert_weights, expert_indices = torch.topk(probs, top_k, dim=-1)  # (B, T, top_k)

    return expert_indices, expert_weights


def compute_expert_utilization(
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Count how often each expert is selected across all tokens.

    Args:
        expert_indices: (B, T, top_k) — indices of selected experts.
        n_experts: total number of experts.

    Returns:
        utilization: (n_experts,) — fraction of total assignments going to each expert.
            Values are in [0, 1] and sum to top_k (one per expert slot per token).
            Each entry is the fraction, so sum of all entries = top_k but individual
            entries are fractions relative to total tokens*top_k assignments.
    """
    B, T, top_k = expert_indices.shape
    total_assignments = B * T * top_k

    # Flatten and count
    flat = expert_indices.reshape(-1)  # (B*T*top_k,)
    counts = torch.zeros(n_experts, dtype=torch.float32, device=expert_indices.device)
    for i in range(n_experts):
        counts[i] = (flat == i).sum().float()

    utilization = counts / total_assignments * top_k  # normalize so sum = top_k, each in [0,1]
    return utilization


def compute_load_balance_score(utilization: torch.Tensor) -> float:
    """Measure how balanced the load is across experts.

    Perfectly balanced = all experts equal = 1/n_experts each.
    Score = 1 - n_experts * std(utilization), clamped to [0, 1].
    Higher score = more balanced.

    Args:
        utilization: (n_experts,) — fraction of assignments per expert.

    Returns:
        float in [0, 1].
    """
    n_experts = utilization.shape[0]
    std = utilization.float().std().item()
    score = 1.0 - n_experts * std
    return float(max(0.0, min(1.0, score)))


def compute_expert_specialization(
    expert_indices: torch.Tensor,
    token_positions: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute how much each expert specializes on certain token positions.

    For each expert, computes the entropy of the position distribution it handles.
    Low entropy = high specialization (expert focuses on few positions).
    Returns normalized specialization scores in [0, 1].

    Args:
        expert_indices: (B, T, top_k) — indices of selected experts.
        token_positions: (T,) — position indices for each token.
        n_experts: total number of experts.

    Returns:
        specialization_scores: (n_experts,) — normalized specialization [0, 1].
            Higher value = more specialized (lower entropy).
    """
    B, T, top_k = expert_indices.shape
    T_pos = token_positions.shape[0]
    log_T = math.log(T_pos) if T_pos > 1 else 1.0

    specialization = torch.zeros(n_experts, dtype=torch.float32)

    for expert_id in range(n_experts):
        # Find all (batch, token, slot) where this expert was selected
        # expert_indices: (B, T, top_k)
        mask = expert_indices == expert_id  # (B, T, top_k)

        # Sum over batch and top_k to get per-position count
        # mask shape (B, T, top_k) -> sum over dim 0 and 2 -> (T,)
        pos_counts = mask.float().sum(dim=0).sum(dim=-1)  # (T,)
        total = pos_counts.sum().item()

        if total > 0:
            p = pos_counts / total  # (T,) probability distribution over positions
            entropy = -(p * torch.log(p + 1e-9)).sum().item()
            normalized_entropy = entropy / log_T
            # specialization = 1 - normalized_entropy (low entropy = high specialization)
            score = 1.0 - normalized_entropy
        else:
            score = 0.0

        specialization[expert_id] = max(0.0, min(1.0, score))

    return specialization


# ---------------------------------------------------------------------------
# MoEAnalyzer class
# ---------------------------------------------------------------------------


class MoEAnalyzer:
    """Analyze MoE routing patterns by hooking into transformer layers.

    For dense models, simulates expert routing via random projection and k-means-style
    token clustering.

    Usage::

        config = MoEAnalysisConfig(n_experts=8, top_k=2)
        analyzer = MoEAnalyzer(model, config)
        handles = analyzer.register_hooks()
        results = analyzer.analyze_batch(input_ids)
        analyzer.remove_hooks(handles)
    """

    def __init__(self, model: Any, config: MoEAnalysisConfig) -> None:
        self.model = model
        self.config = config
        self._layer_hidden_states: dict[int, list[torch.Tensor]] = {}

    def register_hooks(self) -> list:
        """Register forward hooks on model.layers[i] for all tracked layers.

        Returns:
            list of hook handles (call handle.remove() to deregister).
        """
        handles = []

        n_layers = len(self.model.layers)
        layers_to_track = (
            self.config.track_layers if self.config.track_layers else list(range(n_layers))
        )

        for layer_idx in layers_to_track:
            self._layer_hidden_states[layer_idx] = []

            def make_hook(idx: int):
                def hook(module, input, output):
                    # Output may be (hidden, kv) tuple
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._layer_hidden_states[idx].append(hidden.detach())

                return hook

            handle = self.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        return handles

    def remove_hooks(self, handles: list) -> None:
        """Remove all registered hooks.

        Args:
            handles: list of hook handles returned by register_hooks().
        """
        for handle in handles:
            handle.remove()

    def analyze_batch(self, input_ids: torch.Tensor) -> dict[int, dict]:
        """Run one forward pass and analyze MoE routing for each tracked layer.

        Args:
            input_ids: (B, S) token id tensor.

        Returns:
            dict mapping layer_idx -> {
                "utilization": Tensor (n_experts,),
                "load_balance": float,
                "mean_expert_stats": dict,
            }
        """
        # Clear collected states
        for key in self._layer_hidden_states:
            self._layer_hidden_states[key] = []

        # Forward pass — returns (loss, logits, pkv)
        with torch.no_grad():
            _ = self.model(input_ids)

        results: dict[int, dict] = {}

        for layer_idx, hidden_list in self._layer_hidden_states.items():
            if not hidden_list:
                continue

            hidden = hidden_list[-1]  # (B, T, D)
            B, T, D = hidden.shape

            # Simulate routing
            expert_indices, expert_weights = simulate_expert_routing(
                hidden, self.config.n_experts, self.config.top_k
            )

            # Compute utilization
            utilization = compute_expert_utilization(expert_indices, self.config.n_experts)

            # Load balance score
            load_balance = compute_load_balance_score(utilization)

            # Token positions
            token_positions = torch.arange(T, dtype=torch.long)

            # Expert specialization
            specialization = compute_expert_specialization(
                expert_indices, token_positions, self.config.n_experts
            )

            # Per-expert stats
            mean_expert_stats: dict[str, Any] = {}
            for expert_id in range(self.config.n_experts):
                mean_expert_stats[expert_id] = ExpertStats(
                    expert_id=expert_id,
                    utilization=utilization[expert_id].item(),
                    mean_activation=expert_weights[expert_indices == expert_id].mean().item()
                    if (expert_indices == expert_id).any()
                    else 0.0,
                    specialization_score=specialization[expert_id].item(),
                )

            results[layer_idx] = {
                "utilization": utilization,
                "load_balance": load_balance,
                "mean_expert_stats": mean_expert_stats,
            }

        return results

    def run_analysis(self, dataset: list[torch.Tensor]) -> dict:
        """Run analysis over a dataset, aggregating stats across batches.

        Args:
            dataset: list of input_ids tensors, each (B, S).

        Returns:
            dict with keys:
                "mean_load_balance": float — averaged across layers and batches.
                "per_layer_balance": dict[int, float] — per-layer mean load balance.
                "most_used_expert": int — expert with highest mean utilization.
                "least_used_expert": int — expert with lowest mean utilization.
        """
        handles = self.register_hooks()

        per_layer_balances: dict[int, list[float]] = {}
        global_utilization = torch.zeros(self.config.n_experts)
        global_util_count = 0

        try:
            for input_ids in dataset:
                batch_results = self.analyze_batch(input_ids)

                for layer_idx, stats in batch_results.items():
                    if layer_idx not in per_layer_balances:
                        per_layer_balances[layer_idx] = []
                    per_layer_balances[layer_idx].append(stats["load_balance"])

                    global_utilization += stats["utilization"].cpu()
                    global_util_count += 1
        finally:
            self.remove_hooks(handles)

        # Aggregate per-layer
        per_layer_balance: dict[int, float] = {
            layer_idx: sum(vals) / len(vals) for layer_idx, vals in per_layer_balances.items()
        }

        mean_load_balance = (
            sum(per_layer_balance.values()) / len(per_layer_balance) if per_layer_balance else 0.0
        )

        # Global utilization stats
        if global_util_count > 0:
            mean_util = global_utilization / global_util_count
        else:
            mean_util = global_utilization

        most_used_expert = int(mean_util.argmax().item())
        least_used_expert = int(mean_util.argmin().item())

        return {
            "mean_load_balance": float(mean_load_balance),
            "per_layer_balance": per_layer_balance,
            "most_used_expert": most_used_expert,
            "least_used_expert": least_used_expert,
        }
