"""Memory-efficient chunked attention (v3) for the Aurelius LLM.

Implements attention in tiles/chunks to reduce peak memory from O(T^2) to
O(T * chunk_size), remaining numerically equivalent to standard attention.

Pure PyTorch only — no external attention libraries.

Public API:
  - ChunkedAttentionConfig
  - ChunkedSelfAttention
  - ChunkedCrossAttention
  - MemoryUsageEstimator
  - ChunkedAttentionBlock
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
class ChunkedAttentionConfig:
    """Configuration for chunked attention computation."""

    chunk_size: int = 64
    causal: bool = True
    dropout: float = 0.0
    scale: float | None = None  # defaults to 1/sqrt(head_dim) if None


# ---------------------------------------------------------------------------
# ChunkedSelfAttention
# ---------------------------------------------------------------------------


class ChunkedSelfAttention(nn.Module):
    """Memory-efficient self-attention via chunked query computation.

    Reduces peak attention-matrix memory from O(T^2) to O(T * chunk_size)
    by iterating over query chunks and attending each chunk to the full K, V.
    """

    def __init__(self, d_model: int, n_heads: int, config: ChunkedAttentionConfig) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.config = config
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        scale = config.scale
        self._scale: float = scale if scale is not None else math.sqrt(self.d_head)

    # ------------------------------------------------------------------
    # Internal chunked attention kernel
    # ------------------------------------------------------------------

    def _chunked_attn(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """Compute chunked attention.

        Args:
            Q: (B, H, T, D_head)
            K: (B, H, T, D_head)
            V: (B, H, T, D_head)

        Returns:
            (B, H, T, D_head)
        """
        B, H, T, D_head = Q.shape
        chunk_size = self.config.chunk_size
        scale = self._scale

        output_chunks: list[Tensor] = []

        for q_start in range(0, T, chunk_size):
            q_end = min(q_start + chunk_size, T)
            Q_chunk = Q[:, :, q_start:q_end, :]  # (B, H, chunk, D_head)

            # scores: (B, H, chunk, T)
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / scale

            if self.config.causal:
                # Build causal mask: position q_i (absolute) can only attend to k_j where j <= q_i
                q_positions = torch.arange(q_start, q_end, device=Q.device).unsqueeze(
                    1
                )  # (chunk, 1)
                k_positions = torch.arange(T, device=Q.device).unsqueeze(0)  # (1, T)
                # mask[i, j] = True where j > q_positions[i] (should be masked)
                causal_mask = k_positions > q_positions  # (chunk, T)
                # Broadcast over (B, H) dimensions
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.drop(attn_weights)

            # (B, H, chunk, D_head)
            out_chunk = torch.matmul(attn_weights, V)
            output_chunks.append(out_chunk)

        # (B, H, T, D_head)
        return torch.cat(output_chunks, dim=2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Chunked self-attention forward pass.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, H, T, D_head)
        def reshape(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = reshape(Q), reshape(K), reshape(V)

        # (B, H, T, D_head)
        attended = self._chunked_attn(Q, K, V)

        # Reshape back to (B, T, D)
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attended)


# ---------------------------------------------------------------------------
# ChunkedCrossAttention
# ---------------------------------------------------------------------------


class ChunkedCrossAttention(nn.Module):
    """Chunked cross-attention between a query sequence and a context sequence.

    No causal mask — cross-attention is fully non-causal.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // n_heads
        self._scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """Chunked cross-attention forward.

        Args:
            query:   (B, T_q, D)
            context: (B, T_c, D)

        Returns:
            (B, T_q, D)
        """
        B, T_q, D = query.shape
        T_c = context.shape[1]

        Q = self.q_proj(query)  # (B, T_q, D)
        K = self.k_proj(context)  # (B, T_c, D)
        V = self.v_proj(context)  # (B, T_c, D)

        def reshape_q(t: Tensor) -> Tensor:
            return t.view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_q, Dh)

        def reshape_kv(t: Tensor, length: int) -> Tensor:
            return t.view(B, length, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_c, Dh)

        Q = reshape_q(Q)
        K = reshape_kv(K, T_c)
        V = reshape_kv(V, T_c)

        output_chunks: list[Tensor] = []
        for q_start in range(0, T_q, self.chunk_size):
            q_end = min(q_start + self.chunk_size, T_q)
            Q_chunk = Q[:, :, q_start:q_end, :]  # (B, H, chunk, Dh)

            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self._scale  # (B, H, chunk, T_c)
            attn_weights = torch.softmax(scores, dim=-1)
            out_chunk = torch.matmul(attn_weights, V)  # (B, H, chunk, Dh)
            output_chunks.append(out_chunk)

        # (B, H, T_q, Dh) -> (B, T_q, D)
        attended = torch.cat(output_chunks, dim=2)
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, D)

        return self.out_proj(attended)


# ---------------------------------------------------------------------------
# MemoryUsageEstimator
# ---------------------------------------------------------------------------


class MemoryUsageEstimator:
    """Estimate and compare memory usage of standard vs chunked attention."""

    def __init__(self) -> None:
        pass

    def standard_attn_memory(self, B: int, H: int, T: int, D_head: int) -> int:
        """Peak memory for standard attention (bytes, float32).

        Accounts for:
          - Attention matrix:  B * H * T * T * 4
          - Q / K / V tensors: B * H * T * D_head * 4  (per tensor, counted once as representative)
        """
        attn_matrix = B * H * T * T * 4
        qkv = B * H * T * D_head * 4
        return attn_matrix + qkv

    def chunked_attn_memory(self, B: int, H: int, T: int, D_head: int, chunk_size: int) -> int:
        """Peak memory for chunked attention (bytes, float32).

        Only one chunk of the attention matrix materialised at a time:
          - Chunk attention:   B * H * chunk_size * T * 4
          - Q / K / V tensors: B * H * T * D_head * 4
        """
        chunk_attn = B * H * chunk_size * T * 4
        qkv = B * H * T * D_head * 4
        return chunk_attn + qkv

    def memory_reduction(self, B: int, H: int, T: int, D_head: int, chunk_size: int) -> float:
        """Ratio: chunked_memory / standard_memory.

        Returns a value in (0, 1] — equal to 1.0 when chunk_size >= T.
        """
        std = self.standard_attn_memory(B, H, T, D_head)
        chnk = self.chunked_attn_memory(B, H, T, D_head, chunk_size)
        return chnk / std


# ---------------------------------------------------------------------------
# ChunkedAttentionBlock
# ---------------------------------------------------------------------------


class ChunkedAttentionBlock(nn.Module):
    """Full transformer block using chunked self-attention.

    Structure:
        x -> LayerNorm -> ChunkedSelfAttention -> residual
          -> LayerNorm -> FFN (4x hidden, GELU) -> residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 64,
        causal: bool = True,
    ) -> None:
        super().__init__()
        config = ChunkedAttentionConfig(chunk_size=chunk_size, causal=causal)
        self.attn = ChunkedSelfAttention(d_model, n_heads, config)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
