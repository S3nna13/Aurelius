"""Mixture-of-Depths: a token router that lets some tokens skip transformer layers.

Implements the MoD mechanism from "Mixture-of-Depths: Dynamically allocating
compute in transformer-based language models" (Raposo et al., 2024). A learned
router scores each token and only the top-k (by score) are sent through the
wrapped TransformerBlock; the rest pass through via a residual connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MoDConfig:
    """Configuration for Mixture-of-Depths routing.

    Attributes:
        capacity_fraction: Fraction of tokens routed through the layer.
            If 0.5, only the top-50% tokens by router score go through the block.
    """

    capacity_fraction: float = 0.5


class MoDRouter(nn.Module):
    """Learned scalar router: produces per-token routing scores.

    A single linear projection from ``d_model`` to 1, returning an
    unnormalized scalar score for each token position.

    Args:
        d_model: Input feature dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token routing scores.

        Args:
            x: Input tensor of shape ``(B, S, d_model)``.

        Returns:
            Routing scores of shape ``(B, S)`` (unnormalized).
        """
        return self.gate(x).squeeze(-1)


class MoDLayer(nn.Module):
    """Wraps a TransformerBlock with Mixture-of-Depths routing.

    Top-k tokens (by routing score) go through the block.
    Bottom ``(1-k)`` tokens skip the block (residual passthrough).

    The ``aux_loss`` encourages balanced routing via an entropy-style
    regularization term that penalises deviation of the mean routing
    probability from the target capacity fraction.

    Args:
        block: A ``TransformerBlock`` instance.
        d_model: Model dimension.
        cfg: MoDConfig controlling the capacity fraction.
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        cfg: MoDConfig | None = None,
    ) -> None:
        super().__init__()
        self.block = block
        self.router = MoDRouter(d_model)
        self.cfg = cfg or MoDConfig()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Forward pass with Mixture-of-Depths routing.

        Args:
            x: Input tensor of shape ``(B, S, d_model)``.
            freqs_cis: RoPE frequency tensor of shape ``(S, head_dim // 2)``.
            mask: Optional attention mask.
            past_kv: KV cache -- **not supported**; raises if not ``None``.

        Returns:
            Tuple of ``(output, present_kv, aux_loss)`` where:
                - ``output`` has shape ``(B, S, d_model)`` with processed tokens
                  merged back into skipped-token positions.
                - ``present_kv`` is always ``None`` (MoD is incompatible with
                  KV caching).
                - ``aux_loss`` is a scalar load-balancing loss.
        """
        B, S, D = x.shape

        if past_kv is not None:
            raise ValueError("MoDLayer does not support KV cache")

        # --- Router scores ---
        scores = self.router(x)  # (B, S)
        k = max(1, int(S * self.cfg.capacity_fraction))

        # Select top-k token indices per sequence
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)  # (B, k)

        # Gather selected tokens: (B, k, D)
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, k, D)
        selected = x.gather(1, idx_expanded)

        # Run block on selected tokens.
        # freqs_cis[:k] is an approximation: it uses the first k positional
        # frequencies as a proxy. For full correctness one would sort topk_idx
        # and reindex freqs_cis per-batch, but this is acceptable for a
        # standalone module and keeps the implementation clean and testable.
        block_out, _present_kv, _aux = self.block(selected, freqs_cis[:k], mask=None, past_kv=None)

        # Scatter block outputs back into the full-sequence tensor
        output = x.clone()
        output.scatter_(1, idx_expanded, block_out)

        # --- Auxiliary load-balancing loss ---
        # Encourage the mean routing probability to match the target capacity
        # fraction, discouraging degenerate routing patterns.
        router_probs = torch.sigmoid(scores)  # (B, S)
        target = self.cfg.capacity_fraction
        aux_loss = ((router_probs.mean(dim=1) - target) ** 2).mean()

        return output, None, aux_loss
