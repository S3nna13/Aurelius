"""Hierarchical attention for long-document processing.

Processes long sequences by first attending within local chunks, then
computing cross-chunk attention over chunk summaries, and finally
broadcasting global context back to each token.

Architecture:
    1. Chunk input sequence into non-overlapping windows.
    2. Apply local (within-chunk) multi-head attention.
    3. Summarize each chunk to a single vector (mean / first / last).
    4. Apply cross-chunk multi-head attention over the summary sequence.
    5. Broadcast chunk-level context back to every token in each chunk.
    6. Recombine chunks back to original sequence length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HierAttnConfig:
    d_model: int = 512
    n_heads: int = 8
    chunk_size: int = 512
    n_global_tokens: int = 1
    causal: bool = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def chunk_sequence(x: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, int]:
    """Split a (B, T, d) tensor into non-overlapping chunks.

    If T is not divisible by chunk_size the last chunk is zero-padded to
    chunk_size.  The number of padding tokens added is stored as an attribute
    on the returned tensor (``chunked.pad``).

    Args:
        x: Input tensor of shape (B, T, d).
        chunk_size: Number of tokens per chunk.

    Returns:
        Tuple of:
            chunked: Tensor of shape (B * n_chunks, chunk_size, d).
            n_chunks: Number of chunks.
    """
    B, T, d = x.shape
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))  # pad sequence dimension
    T_padded = T + pad
    n_chunks = T_padded // chunk_size
    # (B, n_chunks, chunk_size, d) → (B*n_chunks, chunk_size, d)
    chunked = x.view(B, n_chunks, chunk_size, d).reshape(B * n_chunks, chunk_size, d)
    # Attach padding amount as a plain attribute for downstream use
    chunked.pad = pad  # type: ignore[attr-defined]
    return chunked, n_chunks


def summarize_chunks(
    chunk_outputs: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """Reduce chunk token representations to per-chunk summaries.

    Args:
        chunk_outputs: Tensor of shape (B * n_chunks, chunk_size, d).
        method: One of ``"mean"``, ``"first"``, or ``"last"``.

    Returns:
        Tensor of shape (B, n_chunks, d).  The batch size B and chunk count
        n_chunks are inferred from the calling context; here we return
        (B*n_chunks, d) which the caller must reshape to (B, n_chunks, d).

    Note:
        The caller is responsible for supplying B so that the output can be
        correctly reshaped.  This function returns (B*n_chunks, d); use
        ``.view(B, n_chunks, d)`` after calling.
    """
    if method == "mean":
        summary = chunk_outputs.mean(dim=1)          # (B*n_chunks, d)
    elif method == "first":
        summary = chunk_outputs[:, 0, :]             # (B*n_chunks, d)
    elif method == "last":
        summary = chunk_outputs[:, -1, :]            # (B*n_chunks, d)
    else:
        raise ValueError(f"Unknown summarize method: {method!r}. Use 'mean', 'first', or 'last'.")
    return summary  # (B*n_chunks, d) — caller reshapes to (B, n_chunks, d)


def cross_chunk_attention(
    chunk_summaries: torch.Tensor,
    n_heads: int,
    d_model: int,
) -> torch.Tensor:
    """Apply standard scaled dot-product multi-head attention over chunks.

    Args:
        chunk_summaries: Tensor of shape (B, n_chunks, d_model).
        n_heads: Number of attention heads.
        d_model: Model dimension (must be divisible by n_heads).

    Returns:
        Tensor of shape (B, n_chunks, d_model).
    """
    B, n_chunks, d = chunk_summaries.shape
    assert d == d_model, f"d_model mismatch: got {d}, expected {d_model}"
    head_dim = d_model // n_heads

    # Project Q, K, V inline (no learnable weights — this is a standalone fn).
    # We use F.scaled_dot_product_attention directly with manual reshape.
    # Shape: (B, n_heads, n_chunks, head_dim)
    q = chunk_summaries.view(B, n_chunks, n_heads, head_dim).transpose(1, 2)
    k = chunk_summaries.view(B, n_chunks, n_heads, head_dim).transpose(1, 2)
    v = chunk_summaries.view(B, n_chunks, n_heads, head_dim).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
    # (B, n_heads, n_chunks, head_dim) → (B, n_chunks, d_model)
    out = out.transpose(1, 2).contiguous().view(B, n_chunks, d_model)
    return out


def broadcast_chunk_context(
    chunk_summaries: torch.Tensor,
    chunk_outputs: torch.Tensor,
    n_chunks: int,
) -> torch.Tensor:
    """Add chunk-level summary to every token in the corresponding chunk.

    Args:
        chunk_summaries: Tensor of shape (B, n_chunks, d).
        chunk_outputs: Tensor of shape (B * n_chunks, L, d).
        n_chunks: Number of chunks per batch item.

    Returns:
        Tensor of shape (B * n_chunks, L, d) with context added.
    """
    BNC, L, d = chunk_outputs.shape
    B = BNC // n_chunks
    # (B, n_chunks, d) → (B*n_chunks, 1, d)
    context = chunk_summaries.view(B * n_chunks, 1, d)
    return chunk_outputs + context  # broadcast over L


# ---------------------------------------------------------------------------
# Local (within-chunk) attention
# ---------------------------------------------------------------------------

class LocalAttention(nn.Module):
    """Standard multi-head self-attention with no bias on projections.

    Applied independently to each chunk during the hierarchical forward pass,
    but can also be used stand-alone on a (B, T, d) input.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, T, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Hierarchical attention
# ---------------------------------------------------------------------------

class HierarchicalAttention(nn.Module):
    """Two-level hierarchical attention for long sequences.

    Processing steps:
        1. Chunk input (B, T, d) → (B*n_chunks, chunk_size, d).
        2. Apply ``local_attn`` within each chunk.
        3. Summarize chunks (mean by default) → (B, n_chunks, d).
        4. Apply cross-chunk attention via ``global_proj`` + SDPA.
        5. Broadcast chunk context back to tokens.
        6. Recombine and trim padding to restore (B, T, d).

    Args:
        config: :class:`HierAttnConfig` controlling all hyper-parameters.
    """

    def __init__(self, config: HierAttnConfig) -> None:
        super().__init__()
        self.config = config
        self.local_attn = LocalAttention(config.d_model, config.n_heads)
        self.global_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hierarchical attention forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, d = x.shape
        cfg = self.config

        # --- 1. Chunk ---
        chunked, n_chunks = chunk_sequence(x, cfg.chunk_size)
        pad = chunked.pad  # type: ignore[attr-defined]

        # --- 2. Local attention ---
        local_out = self.local_attn(chunked)  # (B*n_chunks, chunk_size, d)

        # --- 3. Summarize chunks ---
        # summarize_chunks returns (B*n_chunks, d); reshape to (B, n_chunks, d)
        summaries_flat = summarize_chunks(local_out, method="mean")
        summaries = summaries_flat.view(B, n_chunks, d)

        # --- 4. Cross-chunk attention ---
        # Project then attend
        summaries_proj = self.global_proj(summaries)  # (B, n_chunks, d)
        global_ctx = cross_chunk_attention(summaries_proj, cfg.n_heads, cfg.d_model)

        # --- 5. Broadcast back ---
        enriched = broadcast_chunk_context(global_ctx, local_out, n_chunks)
        # (B*n_chunks, chunk_size, d)

        # --- 6. Recombine ---
        T_padded = n_chunks * cfg.chunk_size
        out = enriched.view(B, T_padded, d)
        if pad > 0:
            out = out[:, :T, :]  # strip padding
        return out


# ---------------------------------------------------------------------------
# Hierarchical attention block (with LayerNorm + residual)
# ---------------------------------------------------------------------------

class HierAttnBlock(nn.Module):
    """Pre-norm hierarchical attention block with residual connection.

    Applies LayerNorm → HierarchicalAttention → residual add.

    Args:
        config: :class:`HierAttnConfig` controlling the inner attention.
    """

    def __init__(self, config: HierAttnConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = HierarchicalAttention(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return x + self.attn(self.norm(x))
