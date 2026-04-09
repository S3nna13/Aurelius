"""Expert Choice routing for MoE (Zhou et al., 2022) — experts select top-k tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ExpertChoiceConfig:
    n_experts: int = 8
    capacity_factor: float = 1.25  # each expert processes capacity = capacity_factor * T/n_experts tokens
    d_model: int = 64
    d_ff: int = 128
    use_bias: bool = False


class ExpertChoiceRouter(nn.Module):
    """Router for Expert Choice: each expert selects its top-capacity tokens.

    Args:
        d_model: Hidden dimension.
        n_experts: Number of experts.
    """

    def __init__(self, d_model: int, n_experts: int, capacity_factor: float = 1.25) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.weight = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.weight.weight, std=0.01)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute expert-choice routing.

        Args:
            hidden: (B, T, D)

        Returns:
            indices: (E, capacity) — which N-indices each expert processes
            weights: (E, capacity) — router weights for selected tokens
            router_probs: (N, E) — full softmax (for aux loss)
        """
        B, T, D = hidden.shape
        N = B * T
        hidden_flat = hidden.reshape(N, D)

        # Router logits and softmax
        router_logits = self.weight(hidden_flat)          # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)   # (N, E)

        # Each expert selects top-capacity tokens
        capacity = math.ceil(self.capacity_factor * N / self.n_experts)
        capacity = min(capacity, N)

        # Transpose: (E, N) — each row is one expert's scores over all tokens
        scores = router_probs.T  # (E, N)

        # Top-capacity for each expert
        weights, indices = torch.topk(scores, capacity, dim=-1)  # both (E, capacity)

        return indices, weights, router_probs


class ExpertChoiceFFN(nn.Module):
    """Full MoE FFN using Expert Choice routing.

    Each expert selects exactly capacity = ceil(capacity_factor * N / n_experts) tokens.
    Load balance is guaranteed by construction; aux loss encourages routing entropy.

    Args:
        config: ExpertChoiceConfig
    """

    def __init__(self, config: ExpertChoiceConfig) -> None:
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts

        self.router = ExpertChoiceRouter(
            d_model=config.d_model,
            n_experts=config.n_experts,
            capacity_factor=config.capacity_factor,
        )

        # Each expert: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff, bias=config.use_bias),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_model, bias=config.use_bias),
            )
            for _ in range(config.n_experts)
        ])

        # Store last routing info for token_utilization
        self._last_indices: Tensor | None = None
        self._last_N: int = 0

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Route tokens via Expert Choice and compute weighted output.

        Args:
            hidden: (B, T, D)

        Returns:
            output: (B, T, D)
            aux_loss: scalar auxiliary loss (negative entropy to minimize)
        """
        B, T, D = hidden.shape
        N = B * T
        hidden_flat = hidden.reshape(N, D)

        indices, weights, router_probs = self.router(hidden_flat.reshape(B, T, D))

        # Store for token_utilization
        self._last_indices = indices
        self._last_N = N

        output = torch.zeros_like(hidden_flat)  # (N, D)

        for e, expert in enumerate(self.experts):
            expert_indices = indices[e]                         # (capacity,)
            expert_weights = weights[e]                         # (capacity,)
            expert_input = hidden_flat[expert_indices]          # (capacity, D)
            expert_out = expert(expert_input)                   # (capacity, D)
            # Weighted scatter back
            output[expert_indices] += expert_weights.unsqueeze(-1) * expert_out

        output = output.reshape(B, T, D)
        aux_loss = expert_choice_aux_loss(router_probs)

        return output, aux_loss

    def token_utilization(self) -> dict[str, float]:
        """Returns per-expert token count as fraction of N.

        Must be called after a forward pass.
        """
        if self._last_indices is None:
            return {}
        N = self._last_N
        result = {}
        for e in range(self.n_experts):
            count = self._last_indices[e].numel()
            result[f"expert_{e}"] = count / N
        return result


def expert_choice_aux_loss(router_probs: Tensor) -> Tensor:
    """Entropy regularization on router_probs to encourage token spread.

    Computes negative mean entropy over per-token routing distributions.
    Minimizing this loss maximizes routing entropy (diverse expert usage).

    Args:
        router_probs: (N, E) — softmax probabilities

    Returns:
        scalar: -mean_entropy (minimize to maximize entropy)
    """
    eps = 1e-8
    entropy = -(router_probs * torch.log(router_probs + eps)).sum(dim=-1)  # (N,)
    mean_entropy = entropy.mean()
    return -mean_entropy


class BalancedMoEFFN(nn.Module):
    """Drop-in replacement combining Expert Choice with fallback to dense for short sequences.

    Args:
        config: ExpertChoiceConfig
    """

    def __init__(self, config: ExpertChoiceConfig) -> None:
        super().__init__()
        self.expert_choice = ExpertChoiceFFN(config)
        self.dense_fallback = nn.Linear(config.d_model, config.d_model)
        self.min_tokens_for_moe: int = config.n_experts

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass with fallback to dense for short sequences.

        Args:
            hidden: (B, T, D)

        Returns:
            output: (B, T, D)
            aux_loss: scalar (0.0 for dense path)
        """
        B, T, D = hidden.shape
        N = B * T

        if N < self.min_tokens_for_moe:
            return self.dense_fallback(hidden), torch.zeros(1, device=hidden.device, dtype=hidden.dtype)

        return self.expert_choice(hidden)
