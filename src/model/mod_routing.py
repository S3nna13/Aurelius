"""Mixture-of-Depths with adaptive token routing: dynamically skip transformer layers for unimportant tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoDRoutingConfig:
    """Configuration for adaptive MoD routing.

    Attributes:
        capacity_factor: Fraction of tokens to process per layer (0 < cf < 1).
        n_layers: Total number of transformer layers.
        d_model: Model hidden dimension.
        router_type: "learned" | "attention_based" | "random".
        load_balance_weight: Weight applied to load-balance loss.
        straight_through: Use straight-through estimator for routing gradients.
    """

    capacity_factor: float = 0.5
    n_layers: int = 4
    d_model: int = 512
    router_type: str = "learned"  # "learned" | "attention_based" | "random"
    load_balance_weight: float = 0.01
    straight_through: bool = True


# ---------------------------------------------------------------------------
# Token Importance Router
# ---------------------------------------------------------------------------

class TokenImportanceRouter(nn.Module):
    """Scores token importance for layer-skipping decisions."""

    def __init__(self, d_model: int, router_type: str = "learned") -> None:
        super().__init__()
        self.router_type = router_type

        if router_type == "learned":
            self.scorer = nn.Linear(d_model, 1, bias=False)
        elif router_type == "attention_based":
            self.q = nn.Linear(d_model, d_model // 4)
            self.k = nn.Linear(d_model, d_model // 4)
        elif router_type == "random":
            pass  # no parameters
        else:
            raise ValueError(f"Unknown router_type: {router_type!r}. "
                             "Choose 'learned', 'attention_based', or 'random'.")

    def forward(self, hidden: Tensor) -> Tensor:
        """Compute per-token importance scores.

        Args:
            hidden: (B, T, D) hidden states.

        Returns:
            scores: (B, T) importance scores — higher means more important.
        """
        if self.router_type == "learned":
            # (B, T, 1) -> (B, T)
            return self.scorer(hidden).squeeze(-1)

        elif self.router_type == "attention_based":
            # Self-attention dot-product score: mean over keys
            q = self.q(hidden)  # (B, T, D//4)
            k = self.k(hidden)  # (B, T, D//4)
            # Score each token's query against all keys, reduce to scalar
            scale = q.size(-1) ** -0.5
            # (B, T, T) attention logits; take the mean over the key dimension
            attn = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, T, T)
            scores = attn.mean(dim=-1)  # (B, T)
            return scores

        else:  # random
            B, T, _ = hidden.shape
            return torch.rand(B, T, device=hidden.device, dtype=hidden.dtype)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_tokens(
    hidden: Tensor,
    scores: Tensor,
    capacity: int,
    straight_through: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route the top-k tokens (by score) to be processed; rest are skipped.

    Args:
        hidden: (B, T, D) hidden states.
        scores: (B, T) importance scores.
        capacity: Number of tokens to select (= int(B * T * capacity_factor)).
        straight_through: Whether to apply straight-through estimator.

    Returns:
        selected_hidden: (capacity, D) hidden states of selected tokens.
        selected_indices: (capacity,) long tensor — flat indices in [0, B*T).
        routing_weights: (capacity,) soft weights for gradient flow.
    """
    B, T, D = hidden.shape
    BT = B * T

    # Flatten (B, T) -> (BT,)
    flat_scores = scores.reshape(BT)

    # Clamp capacity to valid range
    capacity = min(capacity, BT)

    # Select top-k indices by score
    topk_vals, topk_indices = torch.topk(flat_scores, k=capacity, sorted=False)

    # Soft routing weights via sigmoid (or softmax over selected)
    routing_weights = torch.sigmoid(topk_vals)  # (capacity,)

    # Gather hidden states for selected positions
    flat_hidden = hidden.reshape(BT, D)
    selected_hidden = flat_hidden[topk_indices]  # (capacity, D)

    if straight_through:
        # Straight-through: forward uses hard selection, backward sees soft weights
        selected_hidden = selected_hidden + (routing_weights.unsqueeze(-1) - routing_weights.unsqueeze(-1).detach())

    return selected_hidden, topk_indices, routing_weights


def scatter_back(
    selected_output: Tensor,
    selected_indices: Tensor,
    routing_weights: Tensor,
    original_hidden: Tensor,
    straight_through: bool = True,
) -> Tensor:
    """Scatter processed tokens back to original positions.

    Non-selected positions retain original_hidden (skip connection).

    Args:
        selected_output: (capacity, D) processed hidden states.
        selected_indices: (capacity,) long — flat indices in [0, B*T).
        routing_weights: (capacity,) soft weights.
        original_hidden: (B, T, D) original hidden states.
        straight_through: Whether to use straight-through estimator.

    Returns:
        output: (B, T, D) with selected positions updated.
    """
    B, T, D = original_hidden.shape
    BT = B * T

    # Start from a copy of original (skip connection for non-selected)
    output = original_hidden.reshape(BT, D).clone()

    # Weight processed outputs
    weighted = selected_output * routing_weights.unsqueeze(-1)  # (capacity, D)

    # Scatter back
    output[selected_indices] = weighted

    return output.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Load balance loss
# ---------------------------------------------------------------------------

def compute_load_balance_loss(scores: Tensor, capacity_factor: float) -> Tensor:
    """Encourage uniform routing across tokens.

    Penalizes the variance of routing probabilities, encouraging each token to
    be selected with probability close to capacity_factor.

    Args:
        scores: (B, T) router logits.
        capacity_factor: Target selection rate per token.

    Returns:
        Scalar loss tensor.
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(scores)  # (B, T)

    # We want each token's routing probability close to capacity_factor
    # Variance of probs measures spread — penalise it
    variance = probs.var()

    # Also add a mean-deviation term: encourage mean(prob) ≈ capacity_factor
    mean_dev = (probs.mean() - capacity_factor) ** 2

    return variance + mean_dev


# ---------------------------------------------------------------------------
# MoD Layer
# ---------------------------------------------------------------------------

class MoDLayer(nn.Module):
    """A transformer layer with MoD token routing."""

    def __init__(self, layer: nn.Module, config: MoDRoutingConfig) -> None:
        super().__init__()
        self.inner_layer = layer
        self.config = config
        self.router = TokenImportanceRouter(config.d_model, config.router_type)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Apply MoD routing then inner layer to selected tokens only.

        Args:
            hidden: (B, T, D) input hidden states.

        Returns:
            output_hidden: (B, T, D) updated hidden states.
            load_balance_loss: scalar tensor.
        """
        B, T, D = hidden.shape
        capacity = max(1, int(B * T * self.config.capacity_factor))

        # 1. Score tokens
        scores = self.router(hidden)  # (B, T)

        # 2. Route top-capacity tokens
        selected_hidden, selected_indices, routing_weights = route_tokens(
            hidden, scores, capacity, self.config.straight_through
        )

        # 3. Apply inner layer to selected tokens only
        processed = self.inner_layer(selected_hidden)  # (capacity, D)

        # 4. Scatter back
        output = scatter_back(
            processed, selected_indices, routing_weights,
            hidden, self.config.straight_through
        )

        # 5. Load balance loss
        lb_loss = compute_load_balance_loss(scores, self.config.capacity_factor)

        return output, lb_loss


# ---------------------------------------------------------------------------
# MoD Transformer
# ---------------------------------------------------------------------------

class MoDTransformer(nn.Module):
    """Stack of MoD layers."""

    def __init__(self, layers: list[nn.Module], config: MoDRoutingConfig) -> None:
        super().__init__()
        self.mod_layers = nn.ModuleList([MoDLayer(l, config) for l in layers])

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Process through all MoD layers, accumulating load balance losses.

        Args:
            hidden: (B, T, D) input hidden states.

        Returns:
            output: (B, T, D) final hidden states.
            total_load_balance_loss: scalar tensor.
        """
        total_lb_loss = torch.zeros(1, device=hidden.device, dtype=hidden.dtype)

        for mod_layer in self.mod_layers:
            hidden, lb_loss = mod_layer(hidden)
            total_lb_loss = total_lb_loss + lb_loss

        return hidden, total_lb_loss.squeeze()


# ---------------------------------------------------------------------------
# Routing pattern analysis
# ---------------------------------------------------------------------------

def analyze_routing_patterns(routing_decisions: list[Tensor]) -> dict[str, float]:
    """Analyze routing decisions across multiple forward steps.

    Args:
        routing_decisions: List of (B, T) boolean tensors indicating which
            tokens were routed (True = processed, False = skipped).

    Returns:
        dict with keys:
            "mean_utilization": Average fraction of tokens routed.
            "routing_variance": Variance of per-token routing rates.
            "consistency": Fraction of tokens always routed or always skipped.
    """
    if not routing_decisions:
        return {"mean_utilization": 0.0, "routing_variance": 0.0, "consistency": 0.0}

    # Stack decisions: (N_steps, B, T)
    stacked = torch.stack([d.float() for d in routing_decisions], dim=0)

    # mean_utilization: average fraction routed per step
    mean_utilization = stacked.mean().item()

    # Per-token routing rate across steps: (B, T)
    per_token_rate = stacked.mean(dim=0)  # (B, T)

    # routing_variance: variance across tokens of their routing rates
    routing_variance = per_token_rate.var().item()

    # consistency: fraction of tokens always routed (rate=1) or always skipped (rate=0)
    always_routed = (per_token_rate == 1.0).float()
    always_skipped = (per_token_rate == 0.0).float()
    consistency = (always_routed + always_skipped).mean().item()

    return {
        "mean_utilization": mean_utilization,
        "routing_variance": routing_variance,
        "consistency": consistency,
    }
