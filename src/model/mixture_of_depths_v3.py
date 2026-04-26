"""Mixture of Depths (Raposo et al. 2024): dynamic token routing.

Only a fraction of tokens (by learned router score) pass through each
transformer block; unselected tokens skip the layer via residual passthrough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MoDConfig:
    """Configuration for Mixture of Depths routing.

    Attributes:
        d_model:  Hidden dimension of the model.
        capacity: Fraction of tokens processed at each MoD block (0 < capacity <= 1).
        n_heads:  Number of attention heads.
        d_ff:     Feed-forward hidden dimension.
    """

    d_model: int = 64
    capacity: float = 0.5
    n_heads: int = 4
    d_ff: int = 128


# ---------------------------------------------------------------------------
# Token Router
# ---------------------------------------------------------------------------


class TokenRouter(nn.Module):
    """Learned per-token scalar router with top-k selection.

    Args:
        d_model:  Input feature dimension.
        capacity: Fraction of tokens to keep (in (0, 1]).
    """

    def __init__(self, d_model: int, capacity: float) -> None:
        super().__init__()
        self.capacity = capacity
        self.linear = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens via top-k selection.

        Args:
            x: Input of shape ``(B, T, d_model)``.

        Returns:
            router_weights:  ``(B, T, 1)`` — sigmoid router scores.
            selected_mask:   ``(B, T)`` bool — True for the top-k tokens.
            aux_loss:        Scalar — load-balancing variance loss.
        """
        B, T, _ = x.shape

        # Compute per-token scores  (B, T, 1)
        router_weights = torch.sigmoid(self.linear(x))  # (B, T, 1)
        scores = router_weights.squeeze(-1)  # (B, T)

        # Top-k selection: keep ceil(capacity * T) tokens per batch item
        k = math.ceil(self.capacity * T)
        k = max(1, min(k, T))  # clamp to [1, T]

        # topk over T dimension → (B, k)
        _, top_indices = torch.topk(scores, k, dim=1)

        # Build boolean mask  (B, T)
        selected_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        selected_mask.scatter_(1, top_indices, True)

        # Aux loss: variance of the mean selection probability across batch items
        # mean score per batch item → (B,); variance encourages load balance
        mean_scores = scores.mean(dim=1)  # (B,)
        aux_loss = mean_scores.var() if B > 1 else mean_scores.new_zeros(())

        return router_weights, selected_mask, aux_loss


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU-style feed-forward block."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# MoDBlock
# ---------------------------------------------------------------------------


class MoDBlock(nn.Module):
    """Single Mixture-of-Depths transformer block.

    Selected tokens pass through pre-norm attention + FFN; unselected tokens
    are copied from the input unchanged (residual shortcut).

    Args:
        config: :class:`MoDConfig` instance.
    """

    def __init__(self, config: MoDConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.router = TokenRouter(d, config.capacity)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_heads,
            batch_first=True,
            bias=False,
        )
        self.ffn = SwiGLUFFN(d, config.d_ff)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply MoD routing and transformer computation.

        Args:
            x: Input of shape ``(B, T, d_model)``.

        Returns:
            output:   Same shape as ``x``.
            aux_loss: Scalar load-balancing loss.
        """
        B, T, D = x.shape

        # 1. Route — get mask for selected tokens
        _, selected_mask, aux_loss = self.router(x)  # mask: (B, T)

        # 2. Initialise output as a copy of the input
        #    (unselected positions are already correct via residual passthrough)
        out = x.clone()

        # 3. Process each batch item's selected tokens independently
        #    We loop over the batch dimension to handle variable-length selections.
        for b in range(B):
            idx = selected_mask[b].nonzero(as_tuple=False).squeeze(1)  # (k,)
            if idx.numel() == 0:
                continue

            # Gather selected tokens  (1, k, D)
            x_sel = x[b, idx, :].unsqueeze(0)

            # Pre-norm attention (self-attention over selected tokens only)
            normed = self.norm1(x_sel)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x_sel = x_sel + attn_out

            # Pre-norm FFN
            x_sel = x_sel + self.ffn(self.norm2(x_sel))

            # Scatter back  (squeeze batch dim)
            out[b, idx, :] = x_sel.squeeze(0)

        return out, aux_loss


# ---------------------------------------------------------------------------
# MoDModel
# ---------------------------------------------------------------------------


class MoDModel(nn.Module):
    """Decoder-only transformer using Mixture-of-Depths blocks.

    Args:
        d_model:    Hidden dimension.
        n_layers:   Number of MoD blocks.
        n_heads:    Number of attention heads.
        d_ff:       FFN hidden dimension.
        capacity:   Fraction of tokens processed per block.
        vocab_size: Vocabulary size.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        capacity: float = 0.5,
        vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        config = MoDConfig(
            d_model=d_model,
            capacity=capacity,
            n_heads=n_heads,
            d_ff=d_ff,
        )
        self.blocks = nn.ModuleList([MoDBlock(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            input_ids: ``(B, T)`` integer token ids.

        Returns:
            logits:         ``(B, T, vocab_size)``.
            total_aux_loss: Scalar sum of auxiliary losses from all blocks.
        """
        x = self.embedding(input_ids)  # (B, T, d_model)

        total_aux_loss = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            total_aux_loss = total_aux_loss + aux

        x = self.norm(x)
        logits = self.unembedding(x)  # (B, T, V)

        return logits, total_aux_loss
