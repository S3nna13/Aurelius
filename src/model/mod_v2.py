"""Mixture-of-Depths v2: Improved MoD with learned routing and capacity tracking.

Implements an improved version of the MoD mechanism from "Mixture-of-Depths:
Dynamically allocating compute in transformer-based language models"
(Raposo et al., 2024).

New features vs. the basic mod.py:
- Per-layer capacity utilization tracking (CapacityTracker)
- Auxiliary load-balancing loss + z-loss regularization
- Top-p routing in addition to top-k and threshold routing
- Routing visualization utilities via CapacityTracker.summary()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig
from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .ffn import SwiGLUFFN
from .rms_norm import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoDv2Config:
    """Configuration for improved Mixture-of-Depths routing.

    Attributes:
        capacity_factor: Fraction of tokens processed per layer (0 < cf < 1).
        routing_type: "top_k" | "top_p" | "threshold".
        router_aux_loss_coeff: Weight for load-balancing auxiliary loss.
        router_z_loss_coeff: Weight for router logit entropy regularization.
        use_aux_loss: Whether to compute and return auxiliary losses.
    """

    capacity_factor: float = 0.5
    routing_type: str = "top_k"          # "top_k" | "top_p" | "threshold"
    router_aux_loss_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001
    use_aux_loss: bool = True


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class RouterV2(nn.Module):
    """Improved MoD router with auxiliary losses.

    Routes tokens to either (a) process through the layer or (b) skip via
    residual connection.

    The router produces a scalar logit per token.  Selected tokens are those
    with the highest router scores (top-k, top-p, or threshold depending on
    ``cfg.routing_type``).  Routing weights are computed as a softmax over the
    *selected* logits so that the weighted combination of routed outputs
    preserves the total "weight mass".

    Two auxiliary losses are computed when ``cfg.use_aux_loss`` is True:
    - load_balance_loss: MSE between mean routing probability and
      ``capacity_factor``, encouraging each layer to route exactly the
      requested fraction of tokens.
    - z_loss: penalises large router logits to prevent logit explosion
      (Zoph et al., 2022 Switch Transformer z-loss formulation).

    Args:
        d_model: Model hidden dimension.
        cfg: MoDv2Config instance.
    """

    def __init__(self, d_model: int, cfg: MoDv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(d_model, 1, bias=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_top_k(
        self, logits: torch.Tensor, capacity: int
    ) -> torch.Tensor:
        """Return boolean mask (B, T) with top-capacity tokens selected."""
        B, T = logits.shape
        # topk returns (values, indices)
        _, top_idx = torch.topk(logits, capacity, dim=1)  # (B, capacity)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return mask

    def _select_top_p(
        self, logits: torch.Tensor, capacity: int
    ) -> torch.Tensor:
        """Return boolean mask selecting tokens via cumulative softmax (top-p).

        Selects the *smallest* set of tokens whose cumulative probability
        (sorted descending) exceeds ``1 - capacity_factor``, but caps at
        ``capacity`` tokens to match the top-k budget.
        """
        B, T = logits.shape
        probs = torch.softmax(logits, dim=-1)  # (B, T)

        # Sort descending to accumulate from most to least probable
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=1)  # (B, T)

        # Keep tokens whose cumulative prob hasn't yet exceeded (1 - p) threshold
        # where p = capacity_factor.  We always keep at least 1 token.
        threshold = 1.0 - self.cfg.capacity_factor
        # A token at position i is "needed" if cumprobs[i-1] < threshold,
        # which is equivalent to: cumprobs shifted right by one < threshold.
        shifted = torch.cat(
            [torch.zeros(B, 1, device=logits.device, dtype=cumprobs.dtype), cumprobs[:, :-1]],
            dim=1,
        )  # (B, T)
        top_p_mask = shifted < threshold  # (B, T) in sorted order

        # Cap to capacity
        position = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, -1)
        top_p_mask = top_p_mask & (position < capacity)

        # Always select at least 1 token
        # (the first position in sorted order is always True after the above)
        top_p_mask[:, 0] = True

        # Scatter back to original token order
        original_mask = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
        original_mask.scatter_(1, sorted_indices, top_p_mask)
        return original_mask

    def _select_threshold(self, logits: torch.Tensor) -> torch.Tensor:
        """Return boolean mask selecting tokens where sigmoid(logit) > 0.5."""
        return torch.sigmoid(logits) > 0.5

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions and auxiliary losses.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Tuple of:
                routing_weights: ``(B, T, 1)`` — soft weights for routed tokens
                    (others are 0).
                route_indices: ``(B, capacity)`` — indices of selected tokens
                    per batch item.  For top-p/threshold, padded to ``capacity``
                    using the top-scoring tokens to ensure consistent shape.
                aux_loss: scalar differentiable auxiliary loss tensor.
        """
        B, T, D = x.shape
        capacity = max(1, math.ceil(T * self.cfg.capacity_factor))

        logits = self.router(x).squeeze(-1)  # (B, T)

        # --- Select tokens ---
        routing_type = self.cfg.routing_type
        if routing_type == "top_k":
            selected_mask = self._select_top_k(logits, capacity)
        elif routing_type == "top_p":
            selected_mask = self._select_top_p(logits, capacity)
        elif routing_type == "threshold":
            selected_mask = self._select_threshold(logits)
        else:
            raise ValueError(
                f"Unknown routing_type={routing_type!r}. "
                "Expected 'top_k', 'top_p', or 'threshold'."
            )

        # --- Build routing_weights: softmax over selected logits ---
        # Mask out non-selected tokens with -inf before softmax
        masked_logits = logits.masked_fill(~selected_mask, float("-inf"))  # (B, T)
        routing_weights = torch.softmax(masked_logits, dim=-1)  # (B, T)
        # Zero out positions that were -inf (softmax gives 0 there but guard anyway)
        routing_weights = routing_weights * selected_mask.float()
        routing_weights = routing_weights.unsqueeze(-1)  # (B, T, 1)

        # --- Build route_indices of shape (B, capacity) ---
        # For top-k the mask selects exactly `capacity` tokens.
        # For top-p/threshold the number of selected tokens can differ;
        # we return the top-`capacity` scoring tokens that are selected, then
        # pad with the top-scoring overall tokens to fill `capacity` slots.
        # We always return exactly `capacity` indices (consistent API).
        _, top_idx = torch.topk(
            logits.masked_fill(~selected_mask, float("-inf")), k=capacity, dim=1
        )  # (B, capacity)
        # For cases where fewer than `capacity` tokens are truly selected (e.g.
        # threshold with very few selected), fall back to top-k from raw logits
        # to fill remaining slots.
        n_selected = selected_mask.sum(dim=1)  # (B,)
        if (n_selected < capacity).any():
            fallback_idx = torch.topk(logits, k=capacity, dim=1).indices  # (B, capacity)
            for b in range(B):
                ns = n_selected[b].item()
                if ns < capacity:
                    top_idx[b] = fallback_idx[b]

        route_indices = top_idx  # (B, capacity)

        # --- Auxiliary losses ---
        if self.cfg.use_aux_loss:
            # Load-balance loss: mean routing prob should be ≈ capacity_factor
            router_probs = torch.sigmoid(logits)  # (B, T)
            mean_prob = router_probs.mean(dim=1)  # (B,)
            load_balance_loss = ((mean_prob - self.cfg.capacity_factor) ** 2).mean()

            # Z-loss: penalise large logits via log(sum(exp(logits)))^2
            log_sum_exp = torch.logsumexp(logits, dim=-1)  # (B,)
            z_loss = (log_sum_exp ** 2).mean()

            aux_loss = (
                self.cfg.router_aux_loss_coeff * load_balance_loss
                + self.cfg.router_z_loss_coeff * z_loss
            )
        else:
            aux_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        return routing_weights, route_indices, aux_loss

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def utilization(self, x: torch.Tensor) -> dict[str, float]:
        """Compute capacity utilization statistics for the given input.

        Args:
            x: Input tensor ``(B, T, D)``.

        Returns:
            Dictionary with keys:
                ``mean_capacity_used``: average number of tokens routed.
                ``fraction_routed``: average fraction of tokens routed.
        """
        B, T, D = x.shape
        capacity = max(1, math.ceil(T * self.cfg.capacity_factor))

        with torch.no_grad():
            logits = self.router(x).squeeze(-1)  # (B, T)

            if self.cfg.routing_type == "top_k":
                selected_mask = self._select_top_k(logits, capacity)
            elif self.cfg.routing_type == "top_p":
                selected_mask = self._select_top_p(logits, capacity)
            elif self.cfg.routing_type == "threshold":
                selected_mask = self._select_threshold(logits)
            else:
                selected_mask = self._select_top_k(logits, capacity)

        n_selected = selected_mask.float().sum(dim=1).mean().item()  # avg over batch
        fraction = n_selected / T
        return {
            "mean_capacity_used": n_selected,
            "fraction_routed": fraction,
        }


# ---------------------------------------------------------------------------
# MoD Layer
# ---------------------------------------------------------------------------

class MoDv2Layer(nn.Module):
    """MoD v2 layer: routes tokens through sublayer, others skip via residual.

    Selected tokens (according to RouterV2) are gathered, processed by
    ``sublayer``, scaled by routing weights, and scattered back into the full
    sequence tensor.  Unrouted tokens are passed through unchanged (pure
    residual — no sublayer computation).

    Args:
        sublayer: Any ``nn.Module`` that maps ``(B', T', D) -> (B', T', D)``.
            For MoD we call it per-batch on the gathered token subset.
        d_model: Model hidden dimension.
        cfg: MoDv2Config.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        d_model: int,
        cfg: MoDv2Config | None = None,
    ) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.router = RouterV2(d_model, cfg or MoDv2Config())
        self.cfg = cfg or MoDv2Config()

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MoD routing and sublayer to input.

        Args:
            x: ``(B, T, D)`` input tensor.
            **kwargs: Additional keyword arguments forwarded to ``sublayer``
                (e.g. ``freqs_cis``, ``mask``).

        Returns:
            Tuple of:
                output: ``(B, T, D)`` — processed output with routing applied.
                aux_loss: scalar differentiable auxiliary loss from the router.
        """
        B, T, D = x.shape
        routing_weights, route_indices, aux_loss = self.router(x)
        # routing_weights: (B, T, 1), route_indices: (B, capacity)

        capacity = route_indices.shape[1]

        # --- Gather selected tokens ---
        # idx_expanded: (B, capacity, D)
        idx_expanded = route_indices.unsqueeze(-1).expand(B, capacity, D)
        x_selected = x.gather(1, idx_expanded)  # (B, capacity, D)

        # --- Process selected tokens through sublayer ---
        # sublayer may return a tuple (e.g. TransformerBlock returns (out, kv))
        # or a plain tensor (e.g. SwiGLUFFN).
        sublayer_out = self.sublayer(x_selected, **kwargs)
        if isinstance(sublayer_out, tuple):
            y_selected = sublayer_out[0]  # (B, capacity, D)
        else:
            y_selected = sublayer_out  # (B, capacity, D)

        # --- Weight selected outputs ---
        # routing_weights_selected: (B, capacity, 1)
        routing_weights_selected = routing_weights.gather(
            1, route_indices.unsqueeze(-1)
        )  # (B, capacity, 1)
        y_selected = y_selected * routing_weights_selected  # broadcast over D

        # --- Scatter back ---
        # Start from x (residual for unrouted tokens).
        output = x.clone()
        output.scatter_(1, idx_expanded, y_selected)

        # Explicitly ensure unrouted tokens are pure residual (no sublayer effect)
        # This is already guaranteed by the scatter into a clone of x, but we
        # reinforce it for clarity and correctness.

        return output, aux_loss


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class MoDv2Transformer(nn.Module):
    """Full decoder-only transformer using MoDv2 routing on every layer.

    Wraps each FFN with a MoDv2Layer; attention always runs on all tokens
    (routing is applied only to the FFN sub-layer, following common practice).

    The forward API matches AureliusTransformer:
        ``(loss_or_none, logits, present_key_values)``

    When ``labels`` are provided, ``loss = ce_loss + total_aux_loss``.
    When only ``input_ids`` are provided, ``loss_or_none = total_aux_loss``
    (a scalar tensor).

    Args:
        config: AureliusConfig.
        mod_cfg: Optional MoDv2Config; defaults to MoDv2Config() if None.
    """

    def __init__(
        self,
        config: AureliusConfig,
        mod_cfg: MoDv2Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mod_cfg = mod_cfg or MoDv2Config()

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Build layers: each block has an attention sub-layer (always runs)
        # and an FFN wrapped in MoDv2Layer.
        self.attn_norms = nn.ModuleList(
            [RMSNorm(config.d_model, eps=config.rms_norm_eps) for _ in range(config.n_layers)]
        )
        self.attns = nn.ModuleList(
            [GroupedQueryAttention(config) for _ in range(config.n_layers)]
        )
        self.ffn_norms = nn.ModuleList(
            [RMSNorm(config.d_model, eps=config.rms_norm_eps) for _ in range(config.n_layers)]
        )
        # Wrap each FFN in a MoDv2Layer
        self.mod_ffns = nn.ModuleList(
            [
                MoDv2Layer(SwiGLUFFN(config), config.d_model, self.mod_cfg)
                for _ in range(config.n_layers)
            ]
        )

        # Final norm + LM head
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies
        freqs = precompute_rope_frequencies(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[Any]]:
        """Forward pass.

        Args:
            input_ids: ``(B, S)`` token indices.
            mask: Optional attention mask.
            labels: ``(B, S)`` target token ids for cross-entropy loss.
            past_key_values: Per-layer KV cache (not supported with MoD).

        Returns:
            ``(loss_or_none, logits, present_key_values)``
        """
        B, S = input_ids.shape

        past_len = (
            past_key_values[0][0].shape[1]
            if past_key_values is not None and past_key_values[0] is not None
            else 0
        )

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[past_len : past_len + S]

        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        present_key_values: list[Any] = []

        for i in range(self.config.n_layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            # Attention sub-layer (all tokens)
            attn_out, kv = self.attns[i](
                self.attn_norms[i](x), freqs_cis, mask, past_kv
            )
            x = x + attn_out
            present_key_values.append(kv)

            # FFN sub-layer (routed via MoDv2)
            # MoDv2Layer's sublayer is SwiGLUFFN which takes (B, T, D) directly.
            ffn_in = self.ffn_norms[i](x)
            ffn_out, aux_loss = self.mod_ffns[i](ffn_in)
            x = x + ffn_out
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            loss: torch.Tensor | None = ce_loss + total_aux_loss
        else:
            loss = total_aux_loss

        return loss, logits, present_key_values


# ---------------------------------------------------------------------------
# Capacity Tracker
# ---------------------------------------------------------------------------

class CapacityTracker:
    """Track per-layer routing statistics across multiple forward passes.

    Usage::

        tracker = CapacityTracker(n_layers)
        # after each forward pass:
        tracker.record(layer_idx, n_routed, n_total)
        stats = tracker.summary()

    Args:
        n_layers: Number of MoD layers to track.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.history: list[list[float]] = [[] for _ in range(n_layers)]

    def record(self, layer_idx: int, n_routed: int, n_total: int) -> None:
        """Record a routing event.

        Args:
            layer_idx: Index of the layer (0-based).
            n_routed: Number of tokens that were routed through the sublayer.
            n_total: Total number of tokens in the sequence.
        """
        if n_total > 0:
            self.history[layer_idx].append(n_routed / n_total)

    def summary(self) -> dict[str, float]:
        """Return per-layer and overall routing statistics.

        Returns:
            Dictionary with keys:
                ``layer_i_mean`` and ``layer_i_std`` for each layer ``i``,
                and ``overall_mean`` across all layers.
        """
        stats: dict[str, float] = {}
        all_fractions: list[float] = []

        for i in range(self.n_layers):
            fracs = self.history[i]
            if fracs:
                mean_val = sum(fracs) / len(fracs)
                variance = sum((f - mean_val) ** 2 for f in fracs) / len(fracs)
                std_val = math.sqrt(variance)
            else:
                mean_val = 0.0
                std_val = 0.0
            stats[f"layer_{i}_mean"] = mean_val
            stats[f"layer_{i}_std"] = std_val
            all_fractions.extend(fracs)

        stats["overall_mean"] = (
            sum(all_fractions) / len(all_fractions) if all_fractions else 0.0
        )
        return stats
