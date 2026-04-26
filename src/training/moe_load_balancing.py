"""MoE load balancing: auxiliary losses to encourage uniform expert utilization.

Implements the Switch Transformer auxiliary loss, z-loss regularization,
top-k routing with capacity constraints, and a full MoE layer with
integrated load balancing.
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
class LoadBalancingConfig:
    """Configuration for MoE load balancing.

    Attributes:
        n_experts:       Total number of experts.
        top_k:           Number of experts each token is routed to.
        aux_loss_coeff:  Weight on the Switch Transformer auxiliary loss.
        z_loss_coeff:    Weight on the router z-loss.
        capacity_factor: Multiplier on the average tokens-per-expert count;
                         controls how much overflow is allowed.
        jitter_eps:      Half-range of uniform noise added to router logits
                         during training (0 disables jitter).
    """

    n_experts: int = 8
    top_k: int = 2
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001
    capacity_factor: float = 1.25
    jitter_eps: float = 0.0


# ---------------------------------------------------------------------------
# Z-loss
# ---------------------------------------------------------------------------


def compute_router_z_loss(router_logits: Tensor) -> Tensor:
    """Z-loss: mean(log(sum(exp(logits)))^2).

    Encourages router logits to stay small, improving numerical stability.

    Args:
        router_logits: (B*T, n_experts) — raw (pre-softmax) router logits.

    Returns:
        Scalar tensor — the z-loss (non-negative).
    """
    # log-sum-exp over experts for each token, then square and average.
    log_z = torch.logsumexp(router_logits, dim=-1)  # (N,)
    z_loss = (log_z**2).mean()
    return z_loss


# ---------------------------------------------------------------------------
# Auxiliary load-balance loss
# ---------------------------------------------------------------------------


def compute_aux_load_balance_loss(
    router_probs: Tensor,
    expert_mask: Tensor,
) -> Tensor:
    """Switch Transformer auxiliary loss.

    L_aux = n_experts * sum_i( f_i * P_i )

    where
        f_i = fraction of tokens routed to expert i  (from expert_mask, discrete)
        P_i = mean router probability for expert i   (from router_probs, differentiable)

    Args:
        router_probs: (B*T, n_experts) — softmax router probabilities.
        expert_mask:  (B*T, n_experts) — one-hot / top-k selection mask.

    Returns:
        Scalar tensor — auxiliary load-balance loss.
    """
    n_experts = router_probs.shape[-1]

    mask_float = expert_mask.float()
    total_selections = mask_float.sum().clamp(min=1.0)
    f_i = mask_float.sum(dim=0) / total_selections  # (E,)

    P_i = router_probs.mean(dim=0)  # (E,)

    return n_experts * (f_i * P_i).sum()


# ---------------------------------------------------------------------------
# Expert utilization statistics
# ---------------------------------------------------------------------------


def compute_expert_utilization(expert_mask: Tensor) -> dict[str, float]:
    """Compute per-expert utilization statistics.

    Args:
        expert_mask: (B*T, n_experts) — selection mask (float or bool).

    Returns:
        Dictionary with keys:
            "utilization_std"  — std of per-expert utilization fractions.
            "min_utilization"  — minimum per-expert utilization fraction.
            "max_utilization"  — maximum per-expert utilization fraction.
            "cv"               — coefficient of variation (std / mean);
                                 lower is more balanced.
    """
    mask_float = expert_mask.float()
    total_selections = mask_float.sum().clamp(min=1.0)
    utilization = mask_float.sum(dim=0) / total_selections  # (E,)

    mean_util = utilization.mean()
    std_util = utilization.std(unbiased=False)
    cv = (std_util / mean_util.clamp(min=1e-9)).item()

    return {
        "utilization_std": std_util.item(),
        "min_utilization": utilization.min().item(),
        "max_utilization": utilization.max().item(),
        "cv": cv,
    }


# ---------------------------------------------------------------------------
# Top-k routing
# ---------------------------------------------------------------------------


def top_k_routing(
    router_logits: Tensor,
    top_k: int,
    capacity_factor: float = 1.25,
    jitter_eps: float = 0.0,
    training: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute top-k routing with optional jitter and capacity constraints.

    Steps:
      1. Optionally add uniform jitter: logits + U[-jitter_eps, jitter_eps]
      2. Compute router_probs = softmax(logits)
      3. top_k_probs, top_k_indices = topk(router_probs, k=top_k)
      4. Normalize top_k_probs to sum to 1 per token
      5. Create expert_mask: (N, n_experts) sparse one-hot over top_k
      6. Compute capacity = int(N / n_experts * capacity_factor * top_k)
         Truncate expert_mask so each expert receives at most capacity tokens

    Args:
        router_logits:   (N, n_experts) — raw router logits.
        top_k:           Number of experts to select per token.
        capacity_factor: Multiplier for per-expert token capacity.
        jitter_eps:      Half-range of uniform noise (0 = no noise).
        training:        If True, apply jitter (when jitter_eps > 0).

    Returns:
        (router_probs, expert_mask, top_k_indices) where
            router_probs:  (N, n_experts) — full softmax probabilities
            expert_mask:   (N, n_experts) — binary mask with capacity truncation
            top_k_indices: (N, top_k)     — which experts each token was routed to
    """
    N, n_experts = router_logits.shape

    # 1. Optional jitter
    if training and jitter_eps > 0.0:
        noise = torch.empty_like(router_logits).uniform_(-jitter_eps, jitter_eps)
        router_logits = router_logits + noise

    # 2. Softmax over experts
    router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

    # 3. Top-k selection
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)
    # top_k_probs: (N, top_k), top_k_indices: (N, top_k)

    # 4. Normalize so each token's selected probabilities sum to 1
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # 5. Build sparse expert_mask: (N, n_experts)
    expert_mask = torch.zeros(N, n_experts, device=router_logits.device, dtype=router_logits.dtype)
    expert_mask.scatter_(1, top_k_indices, 1.0)

    # 6. Capacity constraint: each expert can handle at most `capacity` tokens
    capacity = max(1, int(N / n_experts * capacity_factor * top_k))

    # Process experts in order; for each expert, keep only the first `capacity`
    # tokens (by token index) that were assigned to it.
    for e in range(n_experts):
        col = expert_mask[:, e]  # (N,)
        assigned = col.nonzero(as_tuple=False).squeeze(1)  # token indices
        if assigned.numel() > capacity:
            # Zero out tokens beyond capacity
            overflow = assigned[capacity:]
            expert_mask[overflow, e] = 0.0

    return router_probs, expert_mask, top_k_indices


# ---------------------------------------------------------------------------
# Combine expert outputs
# ---------------------------------------------------------------------------


def combine_expert_outputs(
    expert_outputs: list[Tensor],
    expert_mask: Tensor,
    top_k_probs: Tensor,
    top_k_indices: Tensor,
    N: int,
    D: int,
) -> Tensor:
    """Combine expert outputs weighted by routing probabilities.

    For each token n, sum over its top-k selected experts:
        output[n] = sum_k( top_k_probs[n, k] * expert_outputs[expert_k][pos_in_expert] )

    Args:
        expert_outputs: list of n_experts tensors of shape (capacity, D).
        expert_mask:    (N, n_experts) — binary routing mask.
        top_k_probs:    (N, top_k)    — normalized routing probabilities.
        top_k_indices:  (N, top_k)    — which experts each token uses.
        N:              Number of tokens.
        D:              Model dimension.

    Returns:
        (N, D) combined output tensor.
    """
    n_experts = len(expert_outputs)
    device = expert_mask.device
    dtype = top_k_probs.dtype

    output = torch.zeros(N, D, device=device, dtype=dtype)

    # For each expert, find which tokens were routed there and scatter back
    for e in range(n_experts):
        # Tokens assigned to expert e (capacity-truncated)
        assigned = expert_mask[:, e].nonzero(as_tuple=False).squeeze(1)  # (M,)
        if assigned.numel() == 0:
            continue

        # expert_outputs[e] has shape (capacity, D); take only first M rows
        M = assigned.numel()
        expert_out = expert_outputs[e][:M]  # (M, D)

        # For each token in `assigned`, find its routing weight for expert e
        # by looking up expert e in top_k_indices
        for local_idx, token_idx in enumerate(assigned.tolist()):
            # Find which slot in top_k corresponds to expert e
            token_top_k_experts = top_k_indices[token_idx]  # (top_k,)
            matches = (token_top_k_experts == e).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue
            k_slot = matches[0, 0].item()
            weight = top_k_probs[token_idx, k_slot]
            output[token_idx] += weight * expert_out[local_idx]

    return output


# ---------------------------------------------------------------------------
# LoadBalancedMoELayer
# ---------------------------------------------------------------------------


class LoadBalancedMoELayer(nn.Module):
    """MoE feed-forward layer with integrated load-balancing losses.

    Each token is routed to top_k experts.  The forward pass returns the
    combined output along with a dict of auxiliary losses that the trainer
    should add (weighted) to the task loss.

    Args:
        d_model:    Input/output dimension.
        d_expert:   Hidden dimension of each expert MLP (typically 4 * d_model).
        n_experts:  Number of experts.
        top_k:      Number of experts activated per token.
        cfg:        LoadBalancingConfig controlling loss coefficients etc.
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_experts: int,
        top_k: int,
        cfg: LoadBalancingConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_expert = d_expert
        self.n_experts = n_experts
        self.top_k = top_k
        self.cfg = cfg

        # Router: projects each token to a score over experts
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Experts: each is a 2-layer MLP with ReLU
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_expert),
                    nn.ReLU(),
                    nn.Linear(d_expert, d_model),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass.

        Args:
            x: (B, T, D) input tensor.

        Returns:
            (output, aux_losses) where
                output:     (B, T, D)
                aux_losses: dict with "aux_loss", "z_loss", "total_aux"
        """
        B, T, D = x.shape
        N = B * T

        # Flatten to (N, D) for routing
        x_flat = x.view(N, D)

        # Router logits
        router_logits = self.router(x_flat)  # (N, n_experts)

        # Compute losses on raw logits / full softmax probs
        z_loss = compute_router_z_loss(router_logits)

        # Top-k routing
        router_probs, expert_mask, top_k_indices = top_k_routing(
            router_logits,
            top_k=self.top_k,
            capacity_factor=self.cfg.capacity_factor,
            jitter_eps=self.cfg.jitter_eps,
            training=self.training,
        )

        aux_loss = compute_aux_load_balance_loss(router_probs, expert_mask)

        # Normalized top-k probs (re-extract from router_probs using top_k_indices)
        top_k_probs_raw = router_probs.gather(1, top_k_indices)  # (N, top_k)
        top_k_probs = top_k_probs_raw / top_k_probs_raw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Run each expert on its assigned tokens
        capacity = max(1, int(N / self.n_experts * self.cfg.capacity_factor * self.top_k))
        expert_outputs: list[Tensor] = []
        for e in range(self.n_experts):
            assigned = expert_mask[:, e].nonzero(as_tuple=False).squeeze(1)
            if assigned.numel() == 0:
                expert_out = torch.zeros(capacity, D, device=x.device, dtype=x.dtype)
            else:
                M = assigned.numel()
                tokens = x_flat[assigned]  # (M, D)
                out = self.experts[e](tokens)  # (M, D)
                # Pad to capacity
                if M < capacity:
                    pad = torch.zeros(capacity - M, D, device=x.device, dtype=x.dtype)
                    out = torch.cat([out, pad], dim=0)
                else:
                    out = out[:capacity]
                expert_out = out
            expert_outputs.append(expert_out)

        # Combine outputs
        output_flat = combine_expert_outputs(
            expert_outputs, expert_mask, top_k_probs, top_k_indices, N, D
        )

        output = output_flat.view(B, T, D)

        total_aux = self.cfg.aux_loss_coeff * aux_loss + self.cfg.z_loss_coeff * z_loss

        aux_losses: dict[str, Tensor] = {
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "total_aux": total_aux,
        }

        return output, aux_losses


# ---------------------------------------------------------------------------
# MoELoadBalancer
# ---------------------------------------------------------------------------


class MoELoadBalancer:
    """Training wrapper that tracks and applies MoE load-balancing losses.

    Args:
        model: The nn.Module being trained (may or may not contain MoE layers).
        cfg:   LoadBalancingConfig.
    """

    def __init__(self, model: nn.Module, cfg: LoadBalancingConfig) -> None:
        self.model = model
        self.cfg = cfg

    def compute_combined_loss(
        self,
        task_loss: Tensor,
        aux_losses: list[dict[str, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Combine task loss with auxiliary losses from all MoE layers.

        Args:
            task_loss:  Scalar tensor — primary (e.g., cross-entropy) loss.
            aux_losses: List of aux_loss dicts, one per MoE layer.

        Returns:
            (total_loss, info_dict) where info_dict has keys:
                "task_loss"  — float
                "aux_loss"   — float (sum of weighted aux losses across layers)
                "z_loss"     — float (sum of weighted z losses across layers)
                "total_loss" — float
        """
        total_aux = task_loss.new_zeros(())
        total_aux_unweighted = 0.0
        total_z_unweighted = 0.0

        for layer_aux in aux_losses:
            total_aux = total_aux + layer_aux["total_aux"]
            total_aux_unweighted += layer_aux["aux_loss"].item() * self.cfg.aux_loss_coeff
            total_z_unweighted += layer_aux["z_loss"].item() * self.cfg.z_loss_coeff

        total_loss = task_loss + total_aux

        info: dict[str, float] = {
            "task_loss": task_loss.item(),
            "aux_loss": total_aux_unweighted,
            "z_loss": total_z_unweighted,
            "total_loss": total_loss.item(),
        }

        return total_loss, info

    def get_utilization_report(self, expert_masks: list[Tensor]) -> dict[str, float]:
        """Average utilization statistics across all MoE layers.

        Args:
            expert_masks: List of (B*T, n_experts) masks, one per MoE layer.

        Returns:
            Dict with averaged keys from compute_expert_utilization.
        """
        if not expert_masks:
            return {
                "utilization_std": 0.0,
                "min_utilization": 0.0,
                "max_utilization": 0.0,
                "cv": 0.0,
            }

        aggregated: dict[str, float] = {}
        for mask in expert_masks:
            stats = compute_expert_utilization(mask)
            for k, v in stats.items():
                aggregated[k] = aggregated.get(k, 0.0) + v

        n = len(expert_masks)
        return {k: v / n for k, v in aggregated.items()}
