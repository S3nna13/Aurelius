"""MoE improvements: Switch routing, Z-loss regularization, expert dropout, and token dropping strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MoEImprovementsConfig:
    n_experts: int = 8
    top_k: int = 1                  # Switch Transformer uses top-1
    capacity_factor: float = 1.25   # expert capacity = capacity_factor * tokens / n_experts
    z_loss_coeff: float = 0.001
    expert_dropout: float = 0.0     # randomly disable experts during training
    noise_std: float = 0.01         # jitter noise for router logits
    use_aux_loss: bool = True        # load balancing auxiliary loss


def compute_z_loss(router_logits: Tensor, coeff: float = 0.001) -> Tensor:
    """Z-loss from ST-MoE paper: penalizes large router logits.

    z_loss = coeff * mean(log(sum(exp(router_logits)))^2) per token

    Args:
        router_logits: (B*T, n_experts)
        coeff: regularization coefficient

    Returns:
        scalar z_loss
    """
    # log(sum(exp(logits))) = logsumexp for numerical stability
    log_z = torch.logsumexp(router_logits, dim=-1)  # (B*T,)
    z_loss = coeff * (log_z ** 2).mean()
    return z_loss


def compute_switch_aux_loss(router_probs: Tensor, expert_indices: Tensor) -> Tensor:
    """Switch Transformer auxiliary load balancing loss.

    f_i = fraction of tokens dispatched to expert i
    P_i = mean router probability for expert i
    aux_loss = n_experts * sum(f_i * P_i)

    Args:
        router_probs: (B*T, n_experts) — softmax probabilities
        expert_indices: (B*T,) — selected expert per token (top-1)

    Returns:
        scalar aux_loss
    """
    n_experts = router_probs.shape[-1]
    n_tokens = router_probs.shape[0]

    # f_i: fraction of tokens dispatched to each expert
    one_hot = F.one_hot(expert_indices, num_classes=n_experts).float()  # (B*T, E)
    f_i = one_hot.mean(dim=0)   # (E,)

    # P_i: mean routing probability for each expert
    P_i = router_probs.mean(dim=0)  # (E,)

    aux_loss = n_experts * (f_i * P_i).sum()
    return aux_loss


def add_routing_noise(logits: Tensor, noise_std: float = 0.01, training: bool = True) -> Tensor:
    """Add uniform jitter noise to router logits during training.

    Multiplies logits by (1 + Uniform(0, noise_std)).

    Args:
        logits: router logits of any shape
        noise_std: std / upper bound of uniform noise
        training: only add noise when True

    Returns:
        noisy logits (same shape)
    """
    if not training or noise_std == 0.0:
        return logits
    noise = torch.zeros_like(logits).uniform_(0.0, noise_std)
    return logits * (1.0 + noise)


class SwitchRouter(nn.Module):
    """Switch Transformer top-1 router with capacity-based token dropping.

    Args:
        d_model: hidden dimension
        n_experts: number of experts
        capacity_factor: capacity = int(capacity_factor * B*T / n_experts)
        noise_std: jitter noise std for training
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        capacity_factor: float = 1.25,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        """Compute top-1 routing with capacity constraints.

        Args:
            hidden: (B, T, D)

        Returns:
            router_probs:   (B*T, n_experts) — softmax probabilities
            expert_indices: (B*T,) — selected expert per token
            dispatch_mask:  (B*T,) bool — True = token was dispatched
            overflow_count: int — number of tokens dropped due to capacity
        """
        B, T, D = hidden.shape
        N = B * T
        hidden_flat = hidden.reshape(N, D)

        # Router logits + noise + softmax
        router_logits = self.gate(hidden_flat)  # (N, E)
        router_logits = add_routing_noise(router_logits, self.noise_std, self.training)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        # Top-1 selection
        expert_indices = router_probs.argmax(dim=-1)  # (N,)

        # Compute capacity per expert
        capacity = int(self.capacity_factor * N / self.n_experts)
        capacity = max(capacity, 1)

        # Build dispatch mask: for each expert, keep only first `capacity` tokens
        dispatch_mask = torch.zeros(N, dtype=torch.bool, device=hidden.device)
        overflow_count = 0

        for e in range(self.n_experts):
            token_positions = (expert_indices == e).nonzero(as_tuple=True)[0]  # positions assigned to expert e
            n_assigned = token_positions.shape[0]
            if n_assigned == 0:
                continue
            if n_assigned <= capacity:
                dispatch_mask[token_positions] = True
            else:
                # Keep first `capacity` tokens, drop the rest
                kept = token_positions[:capacity]
                dropped = token_positions[capacity:]
                dispatch_mask[kept] = True
                overflow_count += dropped.shape[0]

        return router_probs, expert_indices, dispatch_mask, overflow_count


class ExpertWithDropout(nn.Module):
    """Expert FFN with expert-level dropout during training.

    Args:
        d_model: hidden dimension
        d_ff: feed-forward intermediate dimension
        expert_dropout: probability of dropping the entire expert output
    """

    def __init__(self, d_model: int, d_ff: int, expert_dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_p = expert_dropout

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Run expert FFN, potentially dropping entire expert output.

        Args:
            x: (N, D) input tokens for this expert
            training: if True, apply expert dropout

        Returns:
            (N, D) output
        """
        if training and self.dropout_p > 0.0:
            # Drop entire expert with prob dropout_p
            if torch.rand(1).item() < self.dropout_p:
                return torch.zeros_like(x)
        return self.ffn(x)


class ImprovedMoELayer(nn.Module):
    """Full MoE layer with Switch routing, Z-loss, and expert dropout.

    Args:
        d_model: hidden dimension
        d_ff: feed-forward intermediate dimension
        config: MoEImprovementsConfig
    """

    def __init__(self, d_model: int, d_ff: int, config: MoEImprovementsConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model

        self.router = SwitchRouter(
            d_model=d_model,
            n_experts=config.n_experts,
            capacity_factor=config.capacity_factor,
            noise_std=config.noise_std,
        )

        self.experts = nn.ModuleList([
            ExpertWithDropout(d_model, d_ff, config.expert_dropout)
            for _ in range(config.n_experts)
        ])

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict]:
        """Route tokens, dispatch to experts, gather outputs.

        Overflow tokens (dropped due to capacity) are copied directly from input.

        Args:
            hidden: (B, T, D)

        Returns:
            output:   (B, T, D)
            aux_dict: {"z_loss": float, "aux_loss": float, "overflow_fraction": float}
        """
        B, T, D = hidden.shape
        N = B * T
        hidden_flat = hidden.reshape(N, D)

        # Routing
        router_probs, expert_indices, dispatch_mask, overflow_count = self.router(hidden)

        # Initialize output — overflow tokens get input copied directly
        output_flat = hidden_flat.clone()

        # Dispatch dispatched tokens to their assigned expert
        for e in range(self.config.n_experts):
            # Tokens assigned to expert e AND not dropped by capacity
            expert_mask = (expert_indices == e) & dispatch_mask  # (N,)
            if not expert_mask.any():
                continue

            expert_tokens = hidden_flat[expert_mask]  # (n, D)
            expert_out = self.experts[e](expert_tokens, training=self.training)  # (n, D)
            output_flat[expert_mask] = expert_out

        output = output_flat.reshape(B, T, D)

        # Compute losses
        router_logits_for_z = self.router.gate(hidden_flat.detach())
        # Use actual gate logits (without noise) for z_loss for stability
        router_logits_actual = self.router.gate(hidden_flat)
        z_loss_val = compute_z_loss(router_logits_actual, self.config.z_loss_coeff)

        aux_loss_val = torch.zeros(1, device=hidden.device, dtype=hidden.dtype).squeeze()
        if self.config.use_aux_loss:
            aux_loss_val = compute_switch_aux_loss(router_probs, expert_indices)

        overflow_fraction = overflow_count / N if N > 0 else 0.0

        aux_dict = {
            "z_loss": z_loss_val,
            "aux_loss": aux_loss_val,
            "overflow_fraction": overflow_fraction,
        }

        return output, aux_dict
