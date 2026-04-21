"""Vision Cross-Attention (Flamingo / LLaVA / Idefics style).

Text hidden states act as *queries*; vision tokens act as *keys* and *values*.
This lets the language model "look up" relevant visual information at each text
position without modifying the vision encoder.

Supports:
- Grouped Query Attention (n_kv_heads < n_heads)
- Optional Flamingo-style tanh gate (output scaled by sigmoid(tanh(gate)))
- Vision padding mask [B, T_vis] — True for valid tokens, False for padding

Algorithm
---------
    Q = text_hidden @ W_Q        [B, T_text, n_heads,    head_dim]
    K = vision_tokens @ W_K      [B, T_vis,  n_kv_heads, head_dim]
    V = vision_tokens @ W_V      [B, T_vis,  n_kv_heads, head_dim]

    attn = softmax(Q @ K^T / sqrt(head_dim))   [B, n_heads, T_text, T_vis]
    out  = attn @ V → reshape → W_O            [B, T_text, d_text]

References
----------
- Flamingo: https://arxiv.org/abs/2204.14198
- LLaVA:    https://arxiv.org/abs/2304.08485
- Idefics:  https://huggingface.co/blog/idefics
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
class VisionCrossAttnConfig:
    """Configuration for VisionCrossAttention.

    Attributes:
        d_text:     Text hidden dimension (query side).
        d_vision:   Vision hidden dimension (key-value side).
        n_heads:    Number of query heads.
        n_kv_heads: Number of key-value heads (GQA when < n_heads).
                    Must divide n_heads evenly.
        head_dim:   Per-head feature dimension.
        dropout:    Attention dropout probability (training only).
        use_gated:  When True, scale output by ``tanh(gate)`` where
                    gate is a learned scalar initialised at 0, giving
                    the model the option to suppress vision at init
                    (Flamingo-style).
    """

    d_text: int = 2048
    d_vision: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout: float = 0.0
    use_gated: bool = False

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )

    @property
    def groups_per_kv(self) -> int:
        """Number of query heads that share each KV head."""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# VisionCrossAttention
# ---------------------------------------------------------------------------


class VisionCrossAttention(nn.Module):
    """Cross-attention from text queries to vision keys / values.

    Text hidden states act as queries; vision tokens act as keys and values.

    Args:
        config: :class:`VisionCrossAttnConfig` instance.
    """

    def __init__(self, config: VisionCrossAttnConfig) -> None:
        super().__init__()
        self.config = config

        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim

        # Query projection from text space
        self.q_proj = nn.Linear(config.d_text, q_dim, bias=False)
        # Key/Value projections from vision space
        self.k_proj = nn.Linear(config.d_vision, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_vision, kv_dim, bias=False)
        # Output projection back to text space
        self.o_proj = nn.Linear(q_dim, config.d_text, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self._scale = math.sqrt(config.head_dim)

        # Optional Flamingo-style tanh gate, initialised at 0
        if config.use_gated:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_attn_weights(
        self,
        text_hidden: Tensor,
        vision_tokens: Tensor,
        vision_mask: Tensor | None,
    ) -> Tensor:
        """Compute scaled dot-product attention weights.

        Args:
            text_hidden:   ``(B, T_text, d_text)``
            vision_tokens: ``(B, T_vis,  d_vision)``
            vision_mask:   ``(B, T_vis)`` — True for valid tokens.

        Returns:
            Attention weights ``(B, n_heads, T_text, T_vis)``.
        """
        cfg = self.config
        B, T_text, _ = text_hidden.shape
        T_vis = vision_tokens.shape[1]

        # Project queries from text
        q = self.q_proj(text_hidden)  # (B, T_text, n_heads * head_dim)
        q = q.view(B, T_text, cfg.n_heads, cfg.head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T_text, head_dim)

        # Project keys/values from vision
        k = self.k_proj(vision_tokens)  # (B, T_vis, n_kv_heads * head_dim)
        k = k.view(B, T_vis, cfg.n_kv_heads, cfg.head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T_vis, head_dim)

        v = self.v_proj(vision_tokens)  # (B, T_vis, n_kv_heads * head_dim)
        v = v.view(B, T_vis, cfg.n_kv_heads, cfg.head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_heads, T_vis, head_dim)

        # GQA: expand KV heads to match query heads
        if cfg.groups_per_kv > 1:
            k = k.repeat_interleave(cfg.groups_per_kv, dim=1)  # (B, n_heads, T_vis, head_dim)
            v = v.repeat_interleave(cfg.groups_per_kv, dim=1)

        # Scaled dot-product scores: (B, n_heads, T_text, T_vis)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self._scale

        # Apply vision padding mask (True = valid, False = padding)
        if vision_mask is not None:
            # vision_mask: (B, T_vis) → (B, 1, 1, T_vis)
            mask = vision_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # If all positions are masked, softmax produces NaN — set to 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        return attn_weights, v

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        text_hidden: Tensor,
        vision_tokens: Tensor,
        vision_mask: Tensor | None = None,
    ) -> Tensor:
        """Run vision cross-attention.

        Args:
            text_hidden:   ``(B, T_text, d_text)``  — text query states.
            vision_tokens: ``(B, T_vis,  d_vision)`` — vision key/value tokens.
            vision_mask:   ``(B, T_vis)``  — bool, True for valid tokens.

        Returns:
            Output tensor ``(B, T_text, d_text)``.
        """
        cfg = self.config
        B, T_text, _ = text_hidden.shape

        attn_weights, v = self._compute_attn_weights(
            text_hidden, vision_tokens, vision_mask
        )

        # Dropout on weights
        attn_weights = self.attn_drop(attn_weights)  # (B, n_heads, T_text, T_vis)

        # Weighted sum of values: (B, n_heads, T_text, head_dim)
        out = torch.matmul(attn_weights, v)

        # Merge heads: (B, T_text, n_heads * head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_text, cfg.n_heads * cfg.head_dim)

        # Flamingo-style tanh gate (zero-init → output ≈ 0 at start)
        if cfg.use_gated and self.gate is not None:
            out = out * torch.tanh(self.gate)

        # Project back to text dimension
        return self.o_proj(out)

    def attention_weights(
        self,
        text_hidden: Tensor,
        vision_tokens: Tensor,
        vision_mask: Tensor | None = None,
    ) -> Tensor:
        """Return attention weights for visualisation.

        Args:
            text_hidden:   ``(B, T_text, d_text)``
            vision_tokens: ``(B, T_vis,  d_vision)``
            vision_mask:   ``(B, T_vis)`` — True for valid tokens.

        Returns:
            Attention weights ``(B, n_heads, T_text, T_vis)``.
        """
        with torch.no_grad():
            weights, _ = self._compute_attn_weights(
                text_hidden, vision_tokens, vision_mask
            )
        return weights
