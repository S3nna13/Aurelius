"""Chunked local attention (Longformer-style).

Each token attends only to tokens within a local sliding window inside its
chunk, reducing full O(N^2) attention to O(N * W). This is a standalone
module that can be swapped in for full attention. Pure PyTorch — no
flash_attn / xformers / einops / transformers.

Reference: Beltagy, Peters, Cohan — "Longformer: The Long-Document Transformer"
(2020), chunked-local component.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedLocalAttention(nn.Module):
    """Chunked local self-attention.

    Sequence is partitioned into chunks of ``chunk_size`` tokens. Within each
    chunk, token at position ``i`` (local index within chunk) attends to
    tokens at local indices ``[max(0, i - window_size), i]``. No cross-chunk
    attention — that is the "chunked" half of Longformer's chunked-local.

    The attention is causal within the window (i.e. only past + current).

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        head_dim: Per-head dimension. Must satisfy ``n_heads * head_dim == d_model``.
        chunk_size: Tokens per chunk. Must be > ``window_size``.
        window_size: Local causal window (inclusive of current token).
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        chunk_size: int = 256,
        window_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be positive int, got {d_model}")
        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError(f"n_heads must be positive int, got {n_heads}")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be positive int, got {head_dim}")
        if n_heads * head_dim != d_model:
            raise ValueError(
                f"n_heads * head_dim ({n_heads} * {head_dim} = {n_heads * head_dim}) "
                f"must equal d_model ({d_model})"
            )
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive int, got {chunk_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be positive int, got {window_size}")
        if chunk_size <= window_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be > window_size ({window_size}); "
                "otherwise degenerates to full attention within the chunk"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    @staticmethod
    def _build_local_mask(chunk_len: int, window_size: int, device, dtype) -> torch.Tensor:
        """Boolean mask of shape (chunk_len, chunk_len); True = attend.

        Entry (i, j) is True iff ``i - window_size <= j <= i``.
        """
        idx = torch.arange(chunk_len, device=device)
        i = idx.unsqueeze(1)  # (L, 1)
        j = idx.unsqueeze(0)  # (1, L)
        allowed = (j <= i) & (j >= i - window_size)
        # Convert to additive mask? SDPA accepts bool directly — True = attend.
        return allowed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, D).

        Returns:
            Tensor of shape (B, S, D), same dtype as input.
        """
        if x.dim() != 3:
            raise ValueError(f"expected 3D input (B, S, D), got shape {tuple(x.shape)}")
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"input last dim {D} != d_model {self.d_model}")

        in_dtype = x.dtype

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Pad sequence to multiple of chunk_size along S.
        C = self.chunk_size
        pad = (C - S % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        S_pad = S + pad
        n_chunks = S_pad // C

        # Reshape into chunks: (B, n_chunks, C, H, Dh) -> (B, n_chunks, H, C, Dh)
        def _chunk(t: torch.Tensor) -> torch.Tensor:
            return (
                t.view(B, n_chunks, C, self.n_heads, self.head_dim)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

        qc = _chunk(q)
        kc = _chunk(k)
        vc = _chunk(v)

        # Flatten (B, n_chunks) for SDPA batch dim: (B*n_chunks, H, C, Dh)
        BN = B * n_chunks
        qc = qc.view(BN, self.n_heads, C, self.head_dim)
        kc = kc.view(BN, self.n_heads, C, self.head_dim)
        vc = vc.view(BN, self.n_heads, C, self.head_dim)

        # Build local causal mask once (C, C) and let SDPA broadcast.
        attn_mask = self._build_local_mask(C, self.window_size, x.device, in_dtype)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            qc,
            kc,
            vc,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        # (BN, H, C, Dh) -> (B, n_chunks, H, C, Dh) -> (B, n_chunks, C, H, Dh) -> (B, S_pad, D)
        out = (
            out.view(B, n_chunks, self.n_heads, C, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        out = out.view(B, S_pad, self.n_heads * self.head_dim)

        if pad:
            out = out[:, :S, :]

        out = self.o_proj(out)
        return out.to(in_dtype)


__all__ = ["ChunkedLocalAttention"]
