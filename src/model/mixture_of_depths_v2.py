"""Mixture of Depths v2: learned top-k token routing with capacity constraints and auxiliary losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoDv2Config:
    """Configuration for Mixture of Depths v2 routing.

    Attributes:
        capacity_factor: Token capacity = capacity_factor * T / n_layers.
        router_z_loss_coeff: Coefficient for z-loss regularization.
        router_aux_loss_coeff: Coefficient for auxiliary load-balance loss.
        top_k: Number of top-k tokens per layer that get full computation.
        use_sigmoid_router: If True use sigmoid routing; else softmax over T.
    """

    capacity_factor: float = 1.25
    router_z_loss_coeff: float = 1e-3
    router_aux_loss_coeff: float = 1e-2
    top_k: int = 1
    use_sigmoid_router: bool = False


# ---------------------------------------------------------------------------
# Token Router
# ---------------------------------------------------------------------------

class TokenRouter(nn.Module):
    """Learned scalar router producing per-token routing scores.

    Args:
        d_model: Input feature dimension.
        use_sigmoid: If True use sigmoid activation; else softmax over T dim.
    """

    def __init__(self, d_model: int, use_sigmoid: bool = False) -> None:
        super().__init__()
        self.router = nn.Linear(d_model, 1, bias=False)
        self.use_sigmoid = use_sigmoid

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute routing weights and logits.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.

        Returns:
            Tuple of:
                routing_weights: ``(B, T, 1)`` — normalized routing weights.
                routing_logits: ``(B, T, 1)`` — raw unnormalized logits.
        """
        routing_logits = self.router(x)  # (B, T, 1)

        if self.use_sigmoid:
            routing_weights = torch.sigmoid(routing_logits)
        else:
            # Softmax over the T dimension
            routing_weights = F.softmax(routing_logits, dim=1)  # (B, T, 1)

        return routing_weights, routing_logits


# ---------------------------------------------------------------------------
# Top-k Token Selection
# ---------------------------------------------------------------------------

def select_top_k_tokens(
    routing_weights: Tensor, k: int, capacity: int
) -> tuple[Tensor, Tensor]:
    """Select top-k tokens by routing weight, capped at capacity.

    Args:
        routing_weights: ``(B, T, 1)`` routing weight tensor.
        k: Number of tokens to select.
        capacity: Maximum number of tokens allowed (capacity constraint).

    Returns:
        Tuple of:
            selected_indices: ``(B, min(k, capacity))`` indices of selected tokens.
            selected_weights: ``(B, min(k, capacity))`` weights of selected tokens.
    """
    B, T, _ = routing_weights.shape
    n_select = min(k, capacity)

    if n_select <= 0:
        # Return empty tensors when k=0 or capacity=0
        selected_indices = torch.zeros(B, 0, dtype=torch.long, device=routing_weights.device)
        selected_weights = torch.zeros(B, 0, device=routing_weights.device)
        return selected_indices, selected_weights

    # Squeeze to (B, T), select top-n_select
    weights_2d = routing_weights.squeeze(-1)  # (B, T)
    top_weights, top_indices = torch.topk(weights_2d, n_select, dim=1)  # (B, n_select)

    return top_indices, top_weights


# ---------------------------------------------------------------------------
# Router Z-loss
# ---------------------------------------------------------------------------

def compute_router_z_loss(routing_logits: Tensor) -> Tensor:
    """Compute z-loss to penalize large routing logits.

    Z-loss = mean(log(sum(exp(logits)))^2)

    Args:
        routing_logits: ``(B, T, 1)`` raw logits from the router.

    Returns:
        Scalar z-loss tensor.
    """
    # routing_logits: (B, T, 1) -> squeeze to (B, T)
    logits_2d = routing_logits.squeeze(-1)  # (B, T)
    log_sum_exp = torch.logsumexp(logits_2d, dim=-1)  # (B,)
    z_loss = (log_sum_exp ** 2).mean()  # scalar
    return z_loss


# ---------------------------------------------------------------------------
# Capacity Utilization
# ---------------------------------------------------------------------------

def compute_capacity_utilization(selected_indices: Tensor, T: int) -> float:
    """Compute fraction of tokens actually processed.

    Args:
        selected_indices: ``(B, n_selected)`` indices of selected tokens.
        T: Total sequence length.

    Returns:
        Fraction in [0, 1] of tokens processed.
    """
    if T == 0:
        return 0.0
    n_selected = selected_indices.shape[1]
    return n_selected / T


# ---------------------------------------------------------------------------
# MoDLayerV2
# ---------------------------------------------------------------------------

class MoDLayerV2(nn.Module):
    """Wraps a layer with Mixture-of-Depths v2 top-k token routing.

    Selected tokens go through the wrapped layer; unselected tokens pass
    through unchanged (identity / residual shortcut).

    Args:
        layer: Any ``nn.Module`` that maps ``(B, n_selected, d_model)`` to same shape.
        d_model: Model hidden dimension.
        config: MoDv2Config instance.
    """

    def __init__(self, layer: nn.Module, d_model: int, config: MoDv2Config) -> None:
        super().__init__()
        self.layer = layer
        self.config = config
        self.token_router = TokenRouter(d_model, use_sigmoid=config.use_sigmoid_router)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Apply MoD v2 routing and layer computation.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            Tuple of:
                x_out: ``(B, T, d_model)`` output tensor.
                info: Dictionary with keys ``router_z_loss`` (Tensor scalar)
                    and ``capacity_utilization`` (float).
        """
        B, T, D = x.shape
        config = self.config

        # Compute capacity (may be fractional — take ceiling, at least 1)
        capacity = max(1, int(config.capacity_factor * T))

        # Route: get weights and logits
        routing_weights, routing_logits = self.token_router(x)  # both (B, T, 1)

        # Compute z-loss
        router_z_loss = compute_router_z_loss(routing_logits)

        # Select top-k tokens
        k = config.top_k
        selected_indices, selected_weights = select_top_k_tokens(routing_weights, k, capacity)
        # selected_indices: (B, n_select), selected_weights: (B, n_select)

        n_select = selected_indices.shape[1]

        # Compute capacity utilization
        cap_util = compute_capacity_utilization(selected_indices, T)

        # Start output as copy of input (unselected tokens = identity)
        x_out = x.clone()

        if n_select > 0:
            # Gather selected tokens: (B, n_select, D)
            idx_expanded = selected_indices.unsqueeze(-1).expand(B, n_select, D)
            x_selected = x.gather(1, idx_expanded)  # (B, n_select, D)

            # Apply layer
            layer_out = self.layer(x_selected)  # (B, n_select, D)
            if isinstance(layer_out, tuple):
                layer_out = layer_out[0]

            # Scatter back into x_out
            x_out.scatter_(1, idx_expanded, layer_out)

        return x_out, {
            "router_z_loss": router_z_loss,
            "capacity_utilization": cap_util,
        }


# ---------------------------------------------------------------------------
# Build MoD v2 Model
# ---------------------------------------------------------------------------

def build_mod_v2_model(
    base_layers: nn.ModuleList,
    d_model: int,
    config: MoDv2Config,
    mod_layer_indices: list[int],
) -> nn.ModuleList:
    """Wrap selected layers with MoDLayerV2, leaving others unchanged.

    Args:
        base_layers: Original list of layers.
        d_model: Model hidden dimension.
        config: MoDv2Config instance.
        mod_layer_indices: List of layer indices to wrap with MoDLayerV2.

    Returns:
        New ModuleList with wrapped and unwrapped layers.
    """
    new_layers = []
    for i, layer in enumerate(base_layers):
        if i in mod_layer_indices:
            new_layers.append(MoDLayerV2(layer, d_model, config))
        else:
            new_layers.append(layer)
    return nn.ModuleList(new_layers)
