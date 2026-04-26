"""Memory-efficient chunked attention (v2) for the Aurelius LLM.

Processes attention in query chunks to reduce peak memory from O(T^2) to
O(T * chunk_size), while remaining numerically equivalent to standard attention.

Exposes the canonical API:
  - ChunkedAttnConfig
  - chunked_attention()
  - ChunkedAttention  (nn.Module, full projection)
  - ChunkedAttnBlock  (LayerNorm + ChunkedAttention + residual)
  - compare_chunked_vs_standard()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkedAttnConfig:
    """Configuration for ChunkedAttention.

    Attributes
    ----------
    d_model:    Model / embedding dimension.
    n_heads:    Number of attention heads. Must divide d_model evenly.
    chunk_size: Query chunk size (controls peak memory).
    causal:     If True, apply causal (autoregressive) masking.
    scale:      Attention scale factor. Defaults to 1/sqrt(d_head).
    """

    d_model: int = 64
    n_heads: int = 4
    chunk_size: int = 64
    causal: bool = True
    scale: float | None = None


# ---------------------------------------------------------------------------
# Core chunked attention function
# ---------------------------------------------------------------------------


def chunked_attention(
    Q: Tensor,  # (B, H, T, d_head)
    K: Tensor,  # (B, H, T, d_head)
    V: Tensor,  # (B, H, T, d_head)
    chunk_size: int,
    causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Compute multi-head attention in query chunks.

    For each query chunk [start : end] (chunk over the T dimension):
      1. Compute scores  = Q_chunk @ K^T * scale  -> (B, H, chunk, T)
      2. Optionally apply causal mask (k_pos > q_pos set to -inf)
      3. Subtract per-row max for numerical stability
      4. Softmax over key dimension
      5. Weighted sum with V

    Concatenate all chunk outputs to produce (B, H, T, d_head), which is
    numerically equivalent to standard scaled-dot-product attention.

    Parameters
    ----------
    Q, K, V:    Query / Key / Value tensors, shape (B, H, T, d_head).
    chunk_size: Number of query positions processed at once.
    causal:     Whether to mask future key positions.
    scale:      Attention scale; defaults to 1/sqrt(d_head).

    Returns
    -------
    Tensor of shape (B, H, T, d_head).
    """
    B, H, T, d_head = Q.shape
    S = K.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    outputs: list[Tensor] = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_chunk = Q[:, :, start:end, :]  # (B, H, chunk, d_head)

        # Attention scores: (B, H, chunk, S)
        scores = torch.matmul(q_chunk, K.transpose(-2, -1)) * scale

        if causal:
            # Query positions in the original sequence: [start, end)
            # Key positions:                            [0, S)
            # Mask out future positions: k_pos > q_pos
            q_pos = torch.arange(start, end, device=Q.device).unsqueeze(1)  # (chunk, 1)
            k_pos = torch.arange(0, S, device=Q.device).unsqueeze(0)  # (1, S)
            causal_mask = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)  # (1, 1, chunk, S)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Numerical stability: subtract per-row maximum
        # Clamp to 0 for fully-masked rows (all -inf) to avoid NaN
        scores_max = scores.amax(dim=-1, keepdim=True)
        scores_max = torch.where(
            torch.isinf(scores_max),
            torch.zeros_like(scores_max),
            scores_max,
        )
        scores = scores - scores_max

        attn = torch.softmax(scores, dim=-1)  # (B, H, chunk, S)
        out_chunk = torch.matmul(attn, V)  # (B, H, chunk, d_head)
        outputs.append(out_chunk)

    return torch.cat(outputs, dim=2)  # (B, H, T, d_head)


# ---------------------------------------------------------------------------
# Reference standard attention (used for testing / comparison)
# ---------------------------------------------------------------------------


def _standard_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Full O(T^2) standard scaled-dot-product attention."""
    B, H, T, d_head = Q.shape
    S = K.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, S)

    if causal:
        q_pos = torch.arange(T, device=Q.device).unsqueeze(1)
        k_pos = torch.arange(S, device=Q.device).unsqueeze(0)
        mask = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


# ---------------------------------------------------------------------------
# ChunkedAttention nn.Module
# ---------------------------------------------------------------------------


class ChunkedAttention(nn.Module):
    """Multi-head attention with chunked query computation.

    Linear projections (no bias) for Q, K, V, and output.
    Input/output shape: (B, T, d_model).
    """

    def __init__(self, config: ChunkedAttnConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (  # noqa: S101
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # No-bias linear projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, H, T, d_head)."""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, d_head) -> (B, T, D)."""
        B, H, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        cfg = self.config
        scale = cfg.scale if cfg.scale is not None else 1.0 / math.sqrt(self.d_head)

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        out = chunked_attention(
            Q,
            K,
            V,
            chunk_size=cfg.chunk_size,
            causal=cfg.causal,
            scale=scale,
        )  # (B, H, T, d_head)

        return self.out_proj(self._merge_heads(out))  # (B, T, d_model)


# ---------------------------------------------------------------------------
# ChunkedAttnBlock: LayerNorm + ChunkedAttention + residual
# ---------------------------------------------------------------------------


class ChunkedAttnBlock(nn.Module):
    """Pre-norm residual block wrapping ChunkedAttention.

    y = x + ChunkedAttention(LayerNorm(x))
    """

    def __init__(self, config: ChunkedAttnConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = ChunkedAttention(config)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        return x + self.attn(self.norm(x))


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_chunked_vs_standard(
    x: Tensor,
    attn: ChunkedAttention,
    chunk_size: int,
) -> float:
    """Run chunked and standard attention, return max absolute difference.

    Parameters
    ----------
    x:          Input tensor (B, T, d_model).
    attn:       ChunkedAttention module (weights shared for both calls).
    chunk_size: Chunk size used for the chunked pass.

    Returns
    -------
    Maximum absolute difference (float) between the two outputs.
    """
    cfg = attn.config
    scale = cfg.scale if cfg.scale is not None else 1.0 / math.sqrt(attn.d_head)

    with torch.no_grad():
        # Chunked path
        Q = attn._split_heads(attn.q_proj(x))
        K = attn._split_heads(attn.k_proj(x))
        V = attn._split_heads(attn.v_proj(x))

        chunked_out = chunked_attention(
            Q, K, V, chunk_size=chunk_size, causal=cfg.causal, scale=scale
        )
        standard_out = _standard_attention(Q, K, V, causal=cfg.causal, scale=scale)

    return (chunked_out - standard_out).abs().max().item()
