"""Mixture-of-Agents (MoA) — ensemble logit fusion.

Multiple model instances are run on the same input; their output logits are
fused via mean, weighted average, or max-probability selection before
decoding.  This is complementary to the text-level MoA (Wang et al. 2024).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoAConfig:
    """Configuration for logit-fusion Mixture-of-Agents.

    Attributes:
        aggregation: How to combine logits. One of "mean", "weighted",
                     "max_prob".
        temperature: Temperature applied to aggregated logits before decoding
                     (>0). A value of 1.0 leaves logits unchanged.
        weights: Per-model weights used when aggregation="weighted".
                 If None and aggregation="weighted", equal weights are used.
    """
    aggregation: str = "mean"
    temperature: float = 1.0
    weights: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_logits_mean(logit_list: List[torch.Tensor]) -> torch.Tensor:
    """Average logits element-wise across models.

    Args:
        logit_list: List of (B, T, V) logit tensors, one per model.

    Returns:
        (B, T, V) tensor of averaged logits.
    """
    stacked = torch.stack(logit_list, dim=0)  # (N, B, T, V)
    return stacked.mean(dim=0)               # (B, T, V)


def aggregate_logits_weighted(
    logit_list: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    """Weighted average of logits: sum(w_i * logits_i) / sum(w_i).

    Args:
        logit_list: List of (B, T, V) logit tensors, one per model.
        weights: Per-model scalar weights (must be same length as logit_list).

    Returns:
        (B, T, V) tensor of weighted-average logits.
    """
    w = torch.tensor(weights, dtype=logit_list[0].dtype,
                     device=logit_list[0].device)  # (N,)
    w_sum = w.sum()
    # Broadcast each weight over (B, T, V)
    result = sum(weights[i] * logit_list[i] for i in range(len(logit_list)))
    return result / w_sum


def aggregate_logits_max_prob(logit_list: List[torch.Tensor]) -> torch.Tensor:
    """At each (batch, position) select logits from the model with the
    highest maximum softmax probability.

    Args:
        logit_list: List of (B, T, V) logit tensors.

    Returns:
        (B, T, V) tensor where each position contains logits from the model
        that had the highest max-probability at that position.
    """
    B, T, V = logit_list[0].shape
    N = len(logit_list)

    stacked = torch.stack(logit_list, dim=0)        # (N, B, T, V)
    probs = F.softmax(stacked, dim=-1)              # (N, B, T, V)
    max_probs = probs.max(dim=-1).values            # (N, B, T)

    # Index of winning model at each (B, T) position
    best_idx = max_probs.argmax(dim=0)              # (B, T)

    # Gather winning logits
    # Expand best_idx to (B, T, V) for indexing into the N dimension
    best_idx_exp = best_idx.unsqueeze(0).unsqueeze(-1).expand(1, B, T, V)  # (1, B, T, V)
    result = stacked.gather(dim=0, index=best_idx_exp).squeeze(0)          # (B, T, V)
    return result


# ---------------------------------------------------------------------------
# MixtureOfAgents
# ---------------------------------------------------------------------------

class MixtureOfAgents:
    """Logit-level ensemble of multiple models.

    Args:
        models: List of model instances.  Each must accept `(input_ids)` and
                return `(loss, logits, past_kv)`.
        config: :class:`MoAConfig` controlling aggregation behaviour.
    """

    def __init__(self, models: list, config: MoAConfig) -> None:
        self.models = models
        self.config = config

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run all models on *input_ids* and return fused logits.

        Args:
            input_ids: (B, T) integer token tensor.

        Returns:
            (B, T, V) aggregated logit tensor.
        """
        logit_list: List[torch.Tensor] = []
        for model in self.models:
            _, logits, _ = model(input_ids)
            logit_list.append(logits)

        agg = self.config.aggregation
        if agg == "mean":
            fused = aggregate_logits_mean(logit_list)
        elif agg == "weighted":
            weights = self.config.weights
            if weights is None:
                weights = [1.0] * len(logit_list)
            fused = aggregate_logits_weighted(logit_list, weights)
        elif agg == "max_prob":
            fused = aggregate_logits_max_prob(logit_list)
        else:
            raise ValueError(f"Unknown aggregation mode: {agg!r}")

        # Apply temperature scaling
        if self.config.temperature != 1.0:
            fused = fused / max(self.config.temperature, 1e-8)

        return fused


# ---------------------------------------------------------------------------
# MoADecoder
# ---------------------------------------------------------------------------

class MoADecoder:
    """Greedy decoder built on top of :class:`MixtureOfAgents`.

    Args:
        models: List of model instances (forwarded to MixtureOfAgents).
        config: :class:`MoAConfig`.
    """

    def __init__(self, models: list, config: MoAConfig) -> None:
        self.moa = MixtureOfAgents(models, config)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Greedy generation using ensemble-aggregated logits.

        Args:
            input_ids: (B, T) prompt token tensor.
            max_new_tokens: Number of new tokens to append.

        Returns:
            (B, T + max_new_tokens) integer token tensor.
        """
        ids = input_ids
        for _ in range(max_new_tokens):
            fused = self.moa.forward(ids)        # (B, T_cur, V)
            next_token = fused[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ids = torch.cat([ids, next_token], dim=1)
        return ids
