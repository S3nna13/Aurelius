"""RETRO — Retrieval-Enhanced Transformer (Borgeaud et al., DeepMind 2021).

Reference: "Improving language models by retrieving from trillions of tokens",
           arXiv:2112.04426

Variable notation follows the paper:
  L       — chunk length (default 64, tiny=8)
  u_i     — i-th input chunk, tokens in [iL, (i+1)L)
  H_i     — hidden states for chunk u_i
  n_i^j   — j-th neighbor chunk retrieved for u_i
  E_i     — encoded neighbor representations for chunk i
  K       — number of neighbors per chunk
  r       — retrieval layer stride (RETRO block every r layers; default r=3)

Architecture (Section 2.2, Figure 1):
  - Input sequence split into chunks of length L
  - Each chunk u_i has K retrieved neighbor chunks from a database
  - Every r-th layer is a RETRO layer: self-attn → CCA(neighbors) → FFN
  - Non-RETRO layers: standard self-attn → FFN
  - RETROEncoder: bidirectional transformer over neighbor tokens
  - Chunked Cross-Attention (CCA): each chunk H_i attends to E_i
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility: scaled dot-product attention (manual, avoids optional deps)
# ---------------------------------------------------------------------------

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention.

    Args:
        q: (B, H, Sq, d_k)
        k: (B, H, Sk, d_k)
        v: (B, H, Sk, d_v)
        attn_mask: optional additive mask broadcastable to (B, H, Sq, Sk)

    Returns:
        (B, H, Sq, d_v)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)
    return torch.matmul(weights, v)


# ---------------------------------------------------------------------------
# RETROEncoder — bidirectional encoder over retrieved neighbors (Section 2.3)
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    """Single bidirectional encoder layer (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm self-attention (bidirectional — no causal mask)
        h = self.norm1(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))
        attn_out = _scaled_dot_product_attention(q, k, v)
        x = x + self.o_proj(self._merge_heads(attn_out))

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class RETROEncoder(nn.Module):
    """Bidirectional encoder E that maps retrieved neighbor chunks to embeddings.

    Given N neighbor sequences each of length neighbor_len, produces the same
    shaped tensor of contextualised representations.

    Input shape:  (B * n_chunks * K, neighbor_len, d_model)
    Output shape: (B * n_chunks * K, neighbor_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B * n_chunks * K, neighbor_len, d_model)
        Returns:
            (B * n_chunks * K, neighbor_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Chunked Cross-Attention (CCA) — Section 2.4
# ---------------------------------------------------------------------------

class RETROCrossChunkAttention(nn.Module):
    """CCA: each input chunk H_i attends to its encoded neighbor representations E_i.

    H_i shape: (B, chunk_size, d_model)  — queries
    E_i shape: (B, K * neighbor_len, d_model)  — keys / values
    Output:    (B, chunk_size, d_model)
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)

    def forward(self, H_i: Tensor, E_i: Tensor) -> Tensor:
        """
        Args:
            H_i: (B, chunk_size, d_model) — chunk hidden states (queries)
            E_i: (B, K * neighbor_len, d_model) — encoded neighbors (keys/values)
        Returns:
            (B, chunk_size, d_model)
        """
        q = self._split_heads(self.q_proj(self.norm_q(H_i)))
        k = self._split_heads(self.k_proj(self.norm_kv(E_i)))
        v = self._split_heads(self.v_proj(self.norm_kv(E_i)))
        out = _scaled_dot_product_attention(q, k, v)
        return self.o_proj(self._merge_heads(out))


# ---------------------------------------------------------------------------
# Standard Transformer Layer (shared by both RETRO and non-RETRO layers)
# ---------------------------------------------------------------------------

class _SelfAttentionLayer(nn.Module):
    """Causal self-attention sub-layer with pre-norm."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)

    def _causal_mask(self, S: int, device: torch.device) -> Tensor:
        mask = torch.full((S, S), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))
        mask = self._causal_mask(x.size(1), x.device)
        out = _scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.o_proj(self._merge_heads(out))


class _FFNLayer(nn.Module):
    """Feed-forward sub-layer with pre-norm."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# RETRO Decoder Layer (RETRO block, Section 2.2)
# ---------------------------------------------------------------------------

class RETROBlock(nn.Module):
    """RETRO decoder layer: self-attn → CCA(neighbors) → FFN.

    Used at every r-th layer. CCA is skipped when neighbors is None
    (falls back to standard self-attn → FFN).

    neighbors shape when provided:
        (B, n_chunks, K, neighbor_len, d_model) — pre-embedded neighbor tokens
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttentionLayer(d_model, n_heads)
        self.cca = RETROCrossChunkAttention(d_model, n_heads)
        self.ffn = _FFNLayer(d_model)

    def forward(
        self,
        x: Tensor,
        neighbors: Optional[Tensor] = None,
        chunk_size: int = 64,
    ) -> Tensor:
        """
        Args:
            x: (B, T, d_model)
            neighbors: optional (B, n_chunks, K, neighbor_len, d_model)
            chunk_size: tokens per chunk L
        Returns:
            (B, T, d_model)
        """
        # 1. Causal self-attention over full sequence
        x = self.self_attn(x)

        # 2. Chunked cross-attention (CCA) — only when neighbors are supplied
        if neighbors is not None:
            B, T, D = x.shape
            n_chunks = neighbors.size(1)
            K = neighbors.size(2)
            neighbor_len = neighbors.size(3)

            # Reshape neighbors: (B, n_chunks, K, neighbor_len, D)
            #   → (B * n_chunks, K * neighbor_len, D)
            E = neighbors.view(B * n_chunks, K * neighbor_len, D)

            # Reshape x into chunks: (B * n_chunks, chunk_size, D)
            # Handle T <= n_chunks * chunk_size (may have been padded)
            padded_T = n_chunks * chunk_size
            if T < padded_T:
                # Pad to full chunk boundary
                pad = torch.zeros(B, padded_T - T, D, device=x.device, dtype=x.dtype)
                x_chunks = torch.cat([x, pad], dim=1)
            else:
                x_chunks = x[:, :padded_T, :]

            H = x_chunks.view(B * n_chunks, chunk_size, D)

            # CCA: H attends to E
            cca_out = self.cca(H, E)           # (B * n_chunks, chunk_size, D)

            # Reshape back and add residual
            cca_out = cca_out.view(B, padded_T, D)
            if T < padded_T:
                cca_out = cca_out[:, :T, :]

            x = x + cca_out

        # 3. FFN
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Standard (non-RETRO) Decoder Layer
# ---------------------------------------------------------------------------

class StandardBlock(nn.Module):
    """Standard transformer decoder layer: self-attn → FFN (no CCA)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttentionLayer(d_model, n_heads)
        self.ffn = _FFNLayer(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.self_attn(x)
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# RETRO Decoder
# ---------------------------------------------------------------------------

class RETRODecoder(nn.Module):
    """RETRO full decoder: interleaved standard and RETRO layers.

    Every layer index in ``retrieval_layers`` is a RETRO block; all others
    are standard transformer layers.

    Args:
        d_model:            hidden dimension
        n_heads:            number of attention heads
        n_layers:           total number of transformer layers
        chunk_size:         L — input chunk length in tokens
        n_retrieval_layers: list of layer indices (0-based) that are RETRO
                            blocks (default: every 3rd layer, i.e. r=3)
        encoder_n_layers:   depth of the neighbor encoder (default 2)

    Forward signature::

        forward(x, neighbors=None) → (B, T, d_model)

    where ``neighbors`` has shape::

        (B, n_chunks, K, neighbor_len, d_model)

    If ``neighbors`` is None the decoder runs as a pure causal transformer
    (no retrieval).  ``d_model``-sized token embeddings must be provided by
    the caller; this module does *not* include an embedding table.

    Raises:
        ValueError: if T is not divisible by chunk_size when neighbors are
                    provided (chunk boundaries must be well-defined).
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        chunk_size: int = 64,
        n_retrieval_layers: Optional[list[int]] = None,
        encoder_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.chunk_size = chunk_size

        # Default: RETRO block at every 3rd layer (r=3), 0-indexed
        if n_retrieval_layers is None:
            n_retrieval_layers = list(range(2, n_layers, 3))  # layers 2,5,8,...
        self.retrieval_layers: set[int] = set(n_retrieval_layers)

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.retrieval_layers:
                self.layers.append(RETROBlock(d_model, n_heads))
            else:
                self.layers.append(StandardBlock(d_model, n_heads))

        self.norm = nn.LayerNorm(d_model)

        # Shared neighbor encoder E (Section 2.3)
        self.encoder = RETROEncoder(d_model, n_heads, n_layers=encoder_n_layers)

    def _encode_neighbors(self, neighbors_raw: Tensor) -> Tensor:
        """Encode raw neighbor token embeddings through RETROEncoder.

        Args:
            neighbors_raw: (B, n_chunks, K, neighbor_len, d_model)
        Returns:
            (B, n_chunks, K, neighbor_len, d_model)  — encoded
        """
        B, n_chunks, K, neighbor_len, D = neighbors_raw.shape
        # Flatten batch × chunks × K for parallel encoding
        flat = neighbors_raw.view(B * n_chunks * K, neighbor_len, D)
        encoded = self.encoder(flat)  # (B*n_chunks*K, neighbor_len, D)
        return encoded.view(B, n_chunks, K, neighbor_len, D)

    def forward(
        self,
        x: Tensor,
        neighbors: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         (B, T, d_model) — token embeddings (pre-embedded by caller)
            neighbors: (B, n_chunks, K, neighbor_len, d_model) — raw neighbor
                       token embeddings, or None for pure-transformer mode

        Returns:
            (B, T, d_model)

        Raises:
            ValueError: if neighbors is provided and T % chunk_size != 0
        """
        B, T, D = x.shape

        # Validate chunk alignment when retrieval is active
        if neighbors is not None:
            if T % self.chunk_size != 0:
                raise ValueError(
                    f"Sequence length T={T} must be divisible by chunk_size="
                    f"{self.chunk_size} when neighbors are provided. "
                    f"Got remainder {T % self.chunk_size}. "
                    "Either pad the input or set neighbors=None."
                )
            # Encode neighbors once before the decoder layers
            neighbors_enc = self._encode_neighbors(neighbors)
        else:
            neighbors_enc = None

        for layer in self.layers:
            if isinstance(layer, RETROBlock):
                x = layer(x, neighbors=neighbors_enc, chunk_size=self.chunk_size)
            else:
                x = layer(x)

        return self.norm(x)
