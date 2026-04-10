"""MoE router load balancing training utilities.

Standalone implementation of Switch Transformer auxiliary loss, z-loss
regularization, expert utilization tracking, and a load-balanced MoE layer.
"""

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
class LoadBalancerConfig:
    """Configuration for MoE load balancer.

    Attributes:
        n_experts:       Total number of experts.
        top_k:           Number of experts each token is routed to.
        aux_loss_coeff:  Weight on the Switch Transformer auxiliary loss.
        z_loss_coeff:    Weight on the router z-loss.
        capacity_factor: Multiplier on the average tokens-per-expert count.
    """

    n_experts: int = 8
    top_k: int = 2
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001
    capacity_factor: float = 1.25


# ---------------------------------------------------------------------------
# Auxiliary load-balance loss (Switch Transformer)
# ---------------------------------------------------------------------------


def compute_aux_loss(router_probs: Tensor, expert_indices: Tensor) -> Tensor:
    """Switch Transformer auxiliary loss encouraging uniform expert utilization.

    L_aux = n_experts * sum_i( f_i * P_i )

    where
        f_i = fraction of tokens routed to expert i  (discrete, from expert_indices)
        P_i = mean router probability for expert i   (differentiable, from router_probs)

    Args:
        router_probs:   (B*T, n_experts) — softmax router probabilities.
        expert_indices: (B*T, top_k)    — indices of selected experts per token.

    Returns:
        Scalar tensor — auxiliary load-balance loss.
    """
    n_tokens, n_experts = router_probs.shape

    # Build one-hot expert mask from indices: (N, n_experts)
    expert_mask = torch.zeros(
        n_tokens, n_experts, device=router_probs.device, dtype=router_probs.dtype
    )
    expert_mask.scatter_(1, expert_indices, 1.0)

    total_selections = expert_mask.sum().clamp(min=1.0)
    f_i = expert_mask.sum(dim=0) / total_selections  # (n_experts,)
    P_i = router_probs.mean(dim=0)                   # (n_experts,)

    return n_experts * (f_i * P_i).sum()


# ---------------------------------------------------------------------------
# Z-loss
# ---------------------------------------------------------------------------


def compute_z_loss(router_logits: Tensor) -> Tensor:
    """Z-loss: mean(log(sum(exp(logits)))^2).

    Stabilizes router logits by discouraging large values.

    Args:
        router_logits: (B*T, n_experts) — raw (pre-softmax) router logits.

    Returns:
        Scalar tensor — z-loss (non-negative).
    """
    log_z = torch.logsumexp(router_logits, dim=-1)  # (N,)
    return (log_z ** 2).mean()


# ---------------------------------------------------------------------------
# Expert utilization
# ---------------------------------------------------------------------------


def compute_expert_utilization(expert_indices: Tensor, n_experts: int) -> Tensor:
    """Compute the fraction of tokens assigned to each expert.

    Args:
        expert_indices: (B*T, top_k) — indices of selected experts per token.
        n_experts:      Total number of experts.

    Returns:
        (n_experts,) tensor of utilization fractions that sum to 1.
    """
    n_tokens = expert_indices.shape[0]
    counts = torch.zeros(n_experts, device=expert_indices.device, dtype=torch.float32)
    # Flatten all expert assignments and count
    flat_indices = expert_indices.reshape(-1)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
    total = counts.sum().clamp(min=1.0)
    return counts / total


# ---------------------------------------------------------------------------
# RouterLinear
# ---------------------------------------------------------------------------


class RouterLinear(nn.Module):
    """Simple linear router that maps token representations to expert scores.

    Args:
        d_model:   Input dimension.
        n_experts: Number of experts to route to.
    """

    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute router probabilities and logits.

        Args:
            x: (..., d_model) input tensor.

        Returns:
            (router_probs, router_logits) both of shape (..., n_experts).
            router_probs = softmax(router_logits).
        """
        router_logits = self.linear(x)                    # (..., n_experts)
        router_probs = F.softmax(router_logits, dim=-1)   # (..., n_experts)
        return router_probs, router_logits


# ---------------------------------------------------------------------------
# LoadBalancedMoELayer
# ---------------------------------------------------------------------------


class LoadBalancedMoELayer(nn.Module):
    """MoE layer with integrated load-balancing losses.

    Each token is routed to top_k experts. The forward pass returns the
    combined output along with a scalar total auxiliary loss.

    Args:
        d_model:   Input/output dimension.
        n_experts: Number of experts.
        d_expert:  Hidden dimension of each expert MLP.
        config:    LoadBalancerConfig controlling loss coefficients.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_expert: int,
        config: LoadBalancerConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.d_expert = d_expert
        self.config = config

        self.router = RouterLinear(d_model, n_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_expert),
                nn.ReLU(),
                nn.Linear(d_expert, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, T, D) or (N, D) input tensor.

        Returns:
            (output, total_aux_loss) where output has the same shape as x
            and total_aux_loss is a scalar.
        """
        original_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
        else:
            x_flat = x
            D = x.shape[-1]

        N = x_flat.shape[0]

        # Route tokens
        router_probs, router_logits = self.router(x_flat)  # (N, E)

        # Compute losses
        z_loss = compute_z_loss(router_logits)

        # Top-k selection
        top_k = self.config.top_k
        top_k_probs_raw, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)
        # (N, top_k)

        # Normalise selected probs
        top_k_probs = top_k_probs_raw / top_k_probs_raw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Auxiliary loss uses full router_probs + discrete expert_indices
        aux_loss = compute_aux_loss(router_probs, top_k_indices)

        # Build expert mask (N, E)
        expert_mask = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        # Compute output: weighted sum of expert outputs
        output_flat = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for e in range(self.n_experts):
            # Tokens assigned to this expert
            assigned = expert_mask[:, e].nonzero(as_tuple=False).squeeze(1)  # (M,)
            if assigned.numel() == 0:
                continue

            tokens = x_flat[assigned]         # (M, D)
            expert_out = self.experts[e](tokens)  # (M, D)

            # Retrieve per-token routing weight for expert e
            # top_k_indices[assigned] is (M, top_k); find slot where == e
            assigned_top_k = top_k_indices[assigned]        # (M, top_k)
            match_mask = (assigned_top_k == e).float()      # (M, top_k)
            weights = (top_k_probs[assigned] * match_mask).sum(dim=-1, keepdim=True)  # (M, 1)

            output_flat[assigned] += weights * expert_out

        total_aux = (
            self.config.aux_loss_coeff * aux_loss
            + self.config.z_loss_coeff * z_loss
        )

        output = output_flat.view(original_shape)
        return output, total_aux


# ---------------------------------------------------------------------------
# LoadBalancerTracker
# ---------------------------------------------------------------------------


class LoadBalancerTracker:
    """Tracks expert utilization statistics across training steps.

    Call update() after each forward pass, get_stats() to retrieve aggregated
    statistics, and reset() to clear accumulated counts.
    """

    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self._counts = torch.zeros(n_experts, dtype=torch.float64)
        self._total = 0.0

    def update(self, expert_indices: Tensor) -> None:
        """Accumulate utilization counts from a batch.

        Args:
            expert_indices: (B*T, top_k) — selected expert indices for a batch.
        """
        flat = expert_indices.reshape(-1).cpu().long()
        counts = torch.zeros(self.n_experts, dtype=torch.float64)
        counts.scatter_add_(0, flat, torch.ones(flat.numel(), dtype=torch.float64))
        self._counts += counts
        self._total += flat.numel()

    def get_stats(self) -> dict:
        """Return aggregated expert utilization statistics.

        Returns:
            Dict with keys:
                "mean_utilization"  — mean fraction across experts
                "max_utilization"   — max fraction across experts
                "min_utilization"   — min fraction across experts
                "utilization_std"   — std of fractions across experts
        """
        total = max(self._total, 1.0)
        utilization = self._counts / total  # (n_experts,)

        return {
            "mean_utilization": float(utilization.mean().item()),
            "max_utilization": float(utilization.max().item()),
            "min_utilization": float(utilization.min().item()),
            "utilization_std": float(utilization.std(correction=0).item()),
        }

    def reset(self) -> None:
        """Clear accumulated utilization statistics."""
        self._counts = torch.zeros(self.n_experts, dtype=torch.float64)
        self._total = 0.0
