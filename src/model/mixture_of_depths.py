"""Mixture of Depths: dynamic token routing through transformer layers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MoDConfig:
    """Configuration for Mixture-of-Depths routing.

    Attributes:
        capacity_factor: Fraction of tokens routed through each layer (k/T).
            E.g. 0.5 means top-50% tokens by router score pass through; rest
            get an identity (residual) pass.
        router_aux_loss_coeff: Coefficient applied to the routing aux loss
            before adding to the total loss.
    """

    capacity_factor: float = 0.5
    router_aux_loss_coeff: float = 0.01


class MoDRouter(nn.Module):
    """Scalar router: scores each token, selects top-k to process.

    A single linear layer maps each token embedding to a scalar score.
    The top-k tokens (by score) are selected for full computation; the rest
    are passed through unchanged.

    Args:
        d_model: Input feature dimension.
        capacity_factor: Fraction of tokens to select per forward pass.
    """

    def __init__(self, d_model: int, capacity_factor: float = 0.5) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.capacity_factor = capacity_factor

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Score tokens and return top-k selection.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.

        Returns:
            Tuple of:
                selected_x: ``(B, k, d_model)`` — top-k token embeddings.
                indices: ``(B, k)`` — positions of selected tokens in [0, T).
                router_scores: ``(B, T)`` — raw scalar scores for all tokens.
        """
        B, T, D = x.shape
        k = max(1, math.ceil(self.capacity_factor * T))

        # Raw scalar scores: (B, T, 1) -> (B, T)
        router_scores = self.gate(x).squeeze(-1)  # (B, T)

        # Select top-k indices per batch element
        _, indices = torch.topk(router_scores, k, dim=1)  # (B, k)

        # Gather selected tokens
        idx_expanded = indices.unsqueeze(-1).expand(B, k, D)
        selected_x = x.gather(1, idx_expanded)  # (B, k, D)

        return selected_x, indices, router_scores


class MoDLayer(nn.Module):
    """Wraps a transformer sub-layer with MoD routing.

    The wrapped ``layer`` must accept a single ``(B, k, d_model)`` tensor and
    return a tensor of the same shape (or a tuple whose first element is that
    tensor). Top-k tokens go through the layer; the rest pass unchanged.

    Args:
        layer: Any ``nn.Module`` with ``forward(x) -> x`` (or ``(x, ...)``).
        d_model: Model hidden dimension.
        config: ``MoDConfig`` controlling capacity and aux-loss coefficient.
    """

    def __init__(self, layer: nn.Module, d_model: int, config: MoDConfig) -> None:
        super().__init__()
        self.layer = layer
        self.router = MoDRouter(d_model, config.capacity_factor)
        self.config = config
        # Track routing fraction across calls for stats
        self._last_tokens_processed_fraction: float = config.capacity_factor

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Route top-k tokens through layer, pass rest unchanged.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            Tuple of:
                output: ``(B, T, d_model)`` — full-sequence output.
                aux_loss: scalar ``Tensor`` — load-balancing auxiliary loss.
        """
        B, T, D = x.shape

        selected_x, indices, router_scores = self.router(x)
        k = selected_x.shape[1]

        # Apply the wrapped layer to the selected tokens
        layer_out = self.layer(selected_x)
        if isinstance(layer_out, tuple):
            layer_out = layer_out[0]  # take first element if layer returns a tuple

        # Scatter processed tokens back into a copy of the full input
        output = x.clone()
        idx_expanded = indices.unsqueeze(-1).expand(B, k, D)
        output.scatter_(1, idx_expanded, layer_out)

        # Compute auxiliary loss
        aux = mod_aux_loss(router_scores, self.config.capacity_factor)

        # Track fraction for stats
        self._last_tokens_processed_fraction = k / T

        return output, aux


def mod_aux_loss(router_scores: Tensor, capacity_factor: float) -> Tensor:
    """Load-balancing auxiliary loss for MoD routers.

    Encourages uniform routing by penalising deviation of the mean sigmoid
    routing probability from the target capacity fraction.

    Args:
        router_scores: ``(B, T)`` raw (unnormalized) scalar scores.
        capacity_factor: Target fraction of tokens to process.

    Returns:
        Non-negative scalar loss tensor.
    """
    router_probs = torch.sigmoid(router_scores)  # (B, T) in (0, 1)
    # Mean routing probability per sequence, then MSE vs target
    mean_prob = router_probs.mean(dim=1)  # (B,)
    loss = ((mean_prob - capacity_factor) ** 2).mean()  # scalar
    return loss


class MoDTransformerWrapper(nn.Module):
    """Wraps an existing ``AureliusTransformer`` model with MoD routing.

    Each layer in ``model.layers`` is replaced by a :class:`MoDLayer` that
    routes only the top-k tokens through the original layer.  The wrapper
    exposes the same ``(loss, logits, past_key_values)`` API as the base model
    and adds the weighted auxiliary routing loss to the returned loss value.

    Args:
        model: An ``AureliusTransformer`` (or any model with ``.layers``
            ``nn.ModuleList`` and ``forward(input_ids) -> (loss, logits, pkv)``).
        config: ``MoDConfig`` controlling capacity and aux-loss coefficient.
    """

    def __init__(self, model: nn.Module, config: MoDConfig) -> None:
        super().__init__()
        self.base_model = model
        self.config = config

        # Determine d_model from the first layer, or fall back to model.config
        d_model = self._infer_d_model(model)

        # Build simple proxy layers that call the original TransformerBlock
        # with dummy freqs_cis/mask — we need a shim because TransformerBlock
        # expects (x, freqs_cis, mask, past_kv) but MoDLayer calls layer(x).
        self.mod_layers = nn.ModuleList()
        for original_layer in model.layers:
            shim = _TransformerBlockShim(original_layer, model)
            mod_layer = MoDLayer(shim, d_model, config)
            self.mod_layers.append(mod_layer)

        # Cumulative aux loss from the last forward pass (for inspection)
        self._last_aux_loss: Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_d_model(model: nn.Module) -> int:
        """Try to read d_model from model.config, else from embed weight."""
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        if hasattr(model, "embed"):
            return model.embed.embedding_dim
        raise ValueError("Cannot infer d_model from model.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Any]:
        """Forward pass identical in signature to the base model.

        Runs the embedding, applies each MoDLayer (which internally calls the
        original transformer block on the top-k tokens), then runs the final
        norm and lm_head.  The accumulated aux loss is added to the
        cross-entropy loss (or returned as the loss if no labels are given).

        Args:
            input_ids: ``(B, S)`` token index tensor.

        Returns:
            Tuple of ``(loss, logits, present_key_values)``:
                - ``loss``: cross-entropy loss + weighted aux loss.  If the
                  base model returns ``None`` loss (no labels), the aux loss
                  alone is returned.
                - ``logits``: ``(B, S, vocab_size)``.
                - ``present_key_values``: empty list (MoD disables KV cache).
        """
        base = self.base_model
        import torch.nn.functional as F

        B, S = input_ids.shape
        x = base.embed(input_ids)

        # Positional frequencies — slice to current seq length
        freqs_cis = base.freqs_cis[:S]

        total_aux = torch.zeros((), device=x.device, dtype=x.dtype)

        for mod_layer in self.mod_layers:
            # Store freqs_cis on the shim so it can pass them to the block
            mod_layer.layer.freqs_cis = freqs_cis  # type: ignore[attr-defined]
            x, aux = mod_layer(x)
            total_aux = total_aux + aux

        x = base.norm(x)
        logits = base.lm_head(x)

        # Scale aux loss
        weighted_aux = self.config.router_aux_loss_coeff * total_aux

        # Build CE loss (no labels — return aux only; else add to CE)
        loss = weighted_aux
        self._last_aux_loss = weighted_aux.detach()

        return loss, logits, []

    # ------------------------------------------------------------------
    # Routing statistics
    # ------------------------------------------------------------------

    def get_routing_stats(self) -> dict:
        """Return routing statistics from the last forward pass.

        Returns:
            Dictionary with keys:
                - ``tokens_processed_fraction``: mean fraction of tokens
                  routed through layers (averaged across all MoD layers).
                - ``capacity_factor``: configured capacity factor.
                - ``n_mod_layers``: number of MoD-wrapped layers.
                - ``aux_loss``: last weighted aux loss value (float).
        """
        fractions = [
            layer._last_tokens_processed_fraction for layer in self.mod_layers
        ]
        mean_fraction = sum(fractions) / len(fractions) if fractions else 0.0
        aux_val = self._last_aux_loss.item() if self._last_aux_loss is not None else 0.0
        return {
            "tokens_processed_fraction": mean_fraction,
            "capacity_factor": self.config.capacity_factor,
            "n_mod_layers": len(self.mod_layers),
            "aux_loss": aux_val,
        }


# ---------------------------------------------------------------------------
# Internal shim: adapts TransformerBlock's 4-arg API to simple forward(x)
# ---------------------------------------------------------------------------

class _TransformerBlockShim(nn.Module):
    """Adapts a ``TransformerBlock`` (which expects freqs_cis/mask/past_kv)
    to the simple ``forward(x) -> x`` API expected by :class:`MoDLayer`.

    ``freqs_cis`` is set by the wrapper before each call.
    """

    def __init__(self, block: nn.Module, model: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.freqs_cis: Tensor | None = None  # set externally before each call

    def forward(self, x: Tensor) -> Tensor:
        if self.freqs_cis is None:
            raise RuntimeError("freqs_cis must be set before calling _TransformerBlockShim.forward")
        T = x.shape[1]
        freqs = self.freqs_cis[:T]
        out, _kv = self.block(x, freqs, mask=None, past_kv=None)
        return out
