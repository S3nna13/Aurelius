"""Adaptive Span Attention for the Aurelius LLM.

Each attention head learns its own attention span — how many past tokens it
attends to.  Heads can specialize: some attend locally (syntactic), others
globally (semantic).  A small L1 penalty on span parameters encourages
shorter spans, reducing effective compute.

Reference: "Adaptive Attention Span in Transformers" (Sukhbaatar et al., 2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSpanConfig:
    d_model: int = 64
    n_heads: int = 4
    max_span: int = 128
    span_loss_coef: float = 0.0002
    init_span: float = 0.5   # initial span as fraction of max_span


# ---------------------------------------------------------------------------
# Differentiable span mask
# ---------------------------------------------------------------------------

def soft_span_mask(span: Tensor, max_span: int, seq_len: int) -> Tensor:
    """Compute a differentiable additive attention mask for adaptive spans.

    Args:
        span:     (n_heads,) float tensor of span fractions in [0, 1].
                  The effective span for head h is span[h] * max_span tokens.
        max_span: maximum number of past tokens any head can attend to.
        seq_len:  sequence length T.

    Returns:
        mask: (n_heads, T, T) additive float mask.
              mask[h, i, j] = 0.0 if j <= i and within span,
              smoothly transitions to -inf at the boundary,
              -inf for j > i (future / causal constraint).

    The ramp function R(x) = clamp((x + 1) / R_size, 0, 1) gates positions
    near the span boundary.  When R(x) = 1 the position is fully attended to
    (additive contribution 0).  When R(x) = 0 it is fully blocked (-inf).

    mask[h, i, j] for j <= i:
        distance d = i - j   (0 == self, positive == past)
        x = span[h] * max_span - d
        R = clamp((x + 1) / R_size, 0, 1)
        additive = 0 if R == 1; log(R) truncated at -1e9 otherwise
    """
    R_size = 4.0
    n_heads = span.shape[0]

    # Build distance matrix: dist[i, j] = i - j  (T x T)
    i_idx = torch.arange(seq_len, device=span.device, dtype=span.dtype).unsqueeze(1)  # (T, 1)
    j_idx = torch.arange(seq_len, device=span.device, dtype=span.dtype).unsqueeze(0)  # (1, T)
    dist = i_idx - j_idx  # (T, T); positive where j <= i, negative where j > i

    # Effective span per head: (n_heads, 1, 1)
    eff_span = (span * max_span).view(n_heads, 1, 1)

    # x = eff_span - dist  →  (n_heads, T, T)
    x = eff_span - dist.unsqueeze(0)  # broadcast over heads

    # Ramp: R = clamp((x + 1) / R_size, 0, 1)
    R = torch.clamp((x + 1.0) / R_size, 0.0, 1.0)  # (n_heads, T, T)

    # Causal mask: future positions (dist < 0, i.e., j > i) → blocked
    causal_ok = dist.unsqueeze(0) >= 0  # (1, T, T) broadcast over heads

    # Additive mask:
    #   causal_ok & R > 0  → log(R)  (0 when R==1)
    #   causal_ok & R == 0 → -inf   (fully blocked)
    #   ~causal_ok         → -inf   (causal constraint)
    NEG_INF = -1e9
    # log(R) where R > 0; clamp to avoid -inf from log(0)
    log_R = torch.clamp(torch.log(R + 1e-10), min=NEG_INF)

    mask = torch.where(causal_ok & (R > 0), log_R, torch.full_like(log_R, NEG_INF))

    return mask  # (n_heads, T, T)


# ---------------------------------------------------------------------------
# AdaptiveSpanAttention module
# ---------------------------------------------------------------------------

class AdaptiveSpanAttention(nn.Module):
    """Multi-head attention where each head learns its own attention span.

    Args:
        config: AdaptiveSpanConfig

    Forward:
        x: (B, T, d_model)
        returns: (output, span_loss)
            output:    (B, T, d_model)
            span_loss: scalar — L1 regularisation on learned spans
    """

    def __init__(self, config: AdaptiveSpanConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.max_span = config.max_span
        self.span_loss_coef = config.span_loss_coef

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learnable span fractions in [0, 1]; shape (n_heads,)
        self.span_params = nn.Parameter(
            torch.full((config.n_heads,), config.init_span)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            output:    (B, T, d_model)
            span_loss: scalar tensor — span_loss_coef * mean(clamped spans)
        """
        B, T, _ = x.shape
        H, d_h = self.n_heads, self.d_head

        # Clamp span params to [0, 1]
        clamped_span = torch.clamp(self.span_params, 0.0, 1.0)  # (H,)

        # Project to Q, K, V and reshape to (B, H, T, d_h)
        Q = self.q_proj(x).view(B, T, H, d_h).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d_h).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, T)
        scale = 1.0 / math.sqrt(d_h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Adaptive span mask: (H, T, T)
        span_mask = soft_span_mask(clamped_span, self.max_span, T)

        # Add mask (broadcasts over batch)
        scores = scores + span_mask.unsqueeze(0)  # (B, H, T, T)

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)  # (B, H, T, d_h)

        out = out.transpose(1, 2).contiguous().view(B, T, H * d_h)
        output = self.out_proj(out)

        # Span regularisation loss
        span_loss = self.span_loss_coef * clamped_span.mean()

        return output, span_loss


# ---------------------------------------------------------------------------
# AdaptiveSpanBlock: pre-norm + AdaptiveSpanAttention + residual
# ---------------------------------------------------------------------------

class AdaptiveSpanBlock(nn.Module):
    """Transformer block: pre-LayerNorm + AdaptiveSpanAttention + residual.

    Forward returns (output, span_loss) so span_loss can be accumulated.
    """

    def __init__(self, config: AdaptiveSpanConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = AdaptiveSpanAttention(config)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            output:    (B, T, d_model) — residual-connected output
            span_loss: scalar tensor
        """
        attn_out, span_loss = self.attn(self.norm(x))
        return x + attn_out, span_loss


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_effective_spans(model: AdaptiveSpanAttention) -> Tensor:
    """Return the effective integer span (in tokens) for each head.

    Args:
        model: AdaptiveSpanAttention instance

    Returns:
        (n_heads,) integer tensor of token spans, bounded by max_span.
    """
    with torch.no_grad():
        clamped = torch.clamp(model.span_params, 0.0, 1.0)
        return (clamped * model.max_span).long()
