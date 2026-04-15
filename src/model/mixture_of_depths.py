"""Mixture of Depths: dynamic token routing through transformer layers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MoDConfig:
    """Configuration for Mixture-of-Depths routing.

    Attributes:
        d_model: Hidden dimension of the model.
        capacity: Fraction of tokens routed through each layer (k/T).
            E.g. 0.125 means top-12.5% of tokens pass through; rest
            get an identity (residual) pass.
        aux_loss_coef: Coefficient applied to the routing auxiliary loss.
    """

    d_model: int = 64
    capacity: float = 0.125
    aux_loss_coef: float = 0.01


def token_router(
    x: Tensor,
    router: nn.Linear,
    capacity: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Score tokens and select the top-capacity fraction for processing.

    Args:
        x: Input tensor of shape ``(B, T, D)``.
        router: A ``Linear(D, 1)`` that produces a scalar score per token.
        capacity: Fraction of tokens to select (k = ceil(capacity * T)).

    Returns:
        Tuple of:
            selected_tokens: ``(B, k, D)`` — top-k token embeddings.
            token_indices:   ``(B, k)``    — positions of selected tokens in [0, T).
            router_weights:  ``(B, k)``    — softmax weights of the selected scores.
    """
    B, T, D = x.shape
    k = max(1, math.ceil(capacity * T))

    # Raw scalar scores: (B, T, 1) -> (B, T)
    scores = router(x).squeeze(-1)  # (B, T)

    # Select top-k indices per batch element
    topk_scores, indices = torch.topk(scores, k, dim=1)  # (B, k)

    # Softmax weights over the selected scores only
    router_weights = torch.softmax(topk_scores, dim=-1)  # (B, k)

    # Gather selected token embeddings
    idx_expanded = indices.unsqueeze(-1).expand(B, k, D)
    selected_tokens = x.gather(1, idx_expanded)  # (B, k, D)

    return selected_tokens, indices, router_weights


def scatter_back(
    output: Tensor,
    indices: Tensor,
    weights: Tensor,
    x: Tensor,
) -> Tensor:
    """Scatter processed tokens back into the full sequence tensor.

    Unselected positions retain their original values from ``x``.
    Selected positions are updated with ``output * weights``.

    Args:
        output:  ``(B, k, D)`` — processed token embeddings.
        indices: ``(B, k)``    — positions of the selected tokens.
        weights: ``(B, k)``    — per-token scalar weights.
        x:       ``(B, T, D)`` — original full-sequence input (provides defaults).

    Returns:
        Full-sequence tensor of shape ``(B, T, D)``.
    """
    B, k, D = output.shape
    result = x.clone()

    # Weight the processed output
    weighted_output = output * weights.unsqueeze(-1)  # (B, k, D)

    # Scatter back
    idx_expanded = indices.unsqueeze(-1).expand(B, k, D)
    result.scatter_(1, idx_expanded, weighted_output)

    return result


def compute_mod_aux_loss(
    router_scores: Tensor,
    capacity: float,
    coef: float,
) -> Tensor:
    """Auxiliary load-balancing loss for MoD routers.

    Encourages uniform routing by penalising deviation of the fraction of
    tokens selected from the target capacity fraction.

    Formula: ``coef * mean((fraction_selected - capacity)^2)`` computed over
    the batch dimension.

    Args:
        router_scores: ``(B, T)`` raw unnormalized scalar scores per token.
        capacity: Target fraction of tokens to process.
        coef: Scalar coefficient to scale the loss.

    Returns:
        Non-negative scalar loss tensor.
    """
    B, T = router_scores.shape
    k = max(1, math.ceil(capacity * T))

    # Compute fraction of tokens "selected" per batch item using a soft proxy:
    # use sigmoid of (score - threshold) to make it differentiable, but for
    # the loss we measure how the mean probability compares to capacity.
    # We use the actual top-k fraction (deterministic) compared to capacity.
    # fraction_selected is always exactly capacity in expectation, so we use
    # the mean sigmoid probability as a soft proxy.
    router_probs = torch.sigmoid(router_scores)  # (B, T), in (0, 1)
    fraction_selected = router_probs.mean(dim=1)  # (B,)

    loss = coef * ((fraction_selected - capacity) ** 2).mean()
    return loss


class MoDLayer(nn.Module):
    """Wraps a transformer sub-layer with Mixture-of-Depths routing.

    The wrapped ``layer`` must accept a ``(B, k, D)`` tensor and return a
    tensor of the same shape (or a tuple whose first element is that tensor).
    Top-k tokens go through the layer; the rest pass unchanged via residual.

    Args:
        layer: Any ``nn.Module`` with ``forward(x) -> x`` (or ``(x, ...)``).
        config: ``MoDConfig`` controlling d_model, capacity, and aux_loss_coef.
    """

    def __init__(self, layer: nn.Module, config: MoDConfig) -> None:
        super().__init__()
        self.layer = layer
        self.config = config
        self.router = nn.Linear(config.d_model, 1, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Route top-k tokens through layer, scatter results back with weights.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            Tuple of:
                output:   ``(B, T, d_model)`` — full-sequence output.
                aux_loss: scalar ``Tensor``    — load-balancing auxiliary loss.
        """
        B, T, D = x.shape

        selected_tokens, token_indices, router_weights = token_router(
            x, self.router, self.config.capacity
        )

        # Pass selected tokens through the wrapped layer
        layer_out = self.layer(selected_tokens)
        if isinstance(layer_out, tuple):
            layer_out = layer_out[0]

        # Scatter processed tokens back; unselected keep original x values
        output = scatter_back(layer_out, token_indices, router_weights, x)

        # Compute auxiliary load-balancing loss
        with torch.no_grad():
            raw_scores = self.router(x).squeeze(-1)  # (B, T)
        # Recompute with grad for the loss
        raw_scores_grad = self.router(x).squeeze(-1)
        aux_loss = compute_mod_aux_loss(raw_scores_grad, self.config.capacity, self.config.aux_loss_coef)

        return output, aux_loss


class MoDTransformer(nn.Module):
    """Wraps a list of transformer layers with Mixture-of-Depths routing.

    Each layer is wrapped in a :class:`MoDLayer`. The forward pass runs
    sequentially through all layers and accumulates auxiliary losses.

    Args:
        layers: List of ``nn.Module`` layers to wrap.
        config: ``MoDConfig`` controlling routing behaviour.
    """

    def __init__(self, layers: List[nn.Module], config: MoDConfig) -> None:
        super().__init__()
        self.config = config
        self.mod_layers = nn.ModuleList(
            MoDLayer(layer, config) for layer in layers
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Sequential forward through all MoDLayers, summing aux losses.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            Tuple of:
                output:         ``(B, T, d_model)`` — final output.
                total_aux_loss: scalar ``Tensor``   — sum of per-layer aux losses.
        """
        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for mod_layer in self.mod_layers:
            x, aux_loss = mod_layer(x)
            total_aux_loss = total_aux_loss + aux_loss
        return x, total_aux_loss
