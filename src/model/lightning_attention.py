"""Lightning Attention-2: tiled linear attention with O(T/B * d²) compute.

Implements Lightning Attention-2 from:
  Qin et al. (2024) "Lightning Attention-2: A Free Lunch for Handling
  Unlimited Sequence Lengths in Large Language Models"
  arXiv:2401.04658

Core idea: causal linear attention is decomposed into
  - Intra-chunk masked matrix multiply (local causal within a block)
  - Inter-chunk recurrent state accumulation (d×d KV state matrix)

This achieves bounded memory and O(T/B * d²) compute for block size B.

Linear attention kernel: φ(x) = ELU(x) + 1  (strictly positive)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Kernel feature map
# ---------------------------------------------------------------------------

def _elu_kernel(x: Tensor) -> Tensor:
    """ELU+1 activation for non-negative linear attention kernel. φ(x) = ELU(x)+1."""
    return F.elu(x) + 1.0


# ---------------------------------------------------------------------------
# 1. LightningLinearAttn — single-head tiled linear attention
# ---------------------------------------------------------------------------

class LightningLinearAttn(nn.Module):
    """Single-head Lightning Attention-2 with tiled linear attention.

    Processes causal linear attention in blocks of size ``chunk_size``.
    Within each chunk the causal part uses a lower-triangular masked matmul.
    Across chunks a running d×d KV state matrix accumulates context.

    Args:
        d_head:     Head dimension.
        chunk_size: Block/tile size B. Can exceed sequence length T.
    """

    def __init__(self, d_head: int, chunk_size: int = 64) -> None:
        super().__init__()
        self.d_head = d_head
        self.chunk_size = chunk_size

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute tiled causal linear attention.

        Args:
            q: (B_batch, T, d_head)
            k: (B_batch, T, d_head)
            v: (B_batch, T, d_head)

        Returns:
            out: (B_batch, T, d_head) — same shape as v.
        """
        B_batch, T, d = q.shape

        # Apply ELU+1 kernel to Q and K for non-negative attention scores.
        q = _elu_kernel(q)
        k = _elu_kernel(k)

        chunk_size = self.chunk_size
        # Running KV outer-product state: (B_batch, d_head, d_head)
        state = torch.zeros(B_batch, d, d, dtype=q.dtype, device=q.device)

        outputs: list[Tensor] = []

        # Pad T to a multiple of chunk_size so we can reshape cleanly.
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            q_padded = F.pad(q, (0, 0, 0, pad_len))
            k_padded = F.pad(k, (0, 0, 0, pad_len))
            v_padded = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_padded, k_padded, v_padded = q, k, v

        T_padded = T + pad_len
        n_chunks = T_padded // chunk_size

        # Reshape to (B_batch, n_chunks, chunk_size, d_head)
        Q_c = q_padded.view(B_batch, n_chunks, chunk_size, d)
        K_c = k_padded.view(B_batch, n_chunks, chunk_size, d)
        V_c = v_padded.view(B_batch, n_chunks, chunk_size, d)

        # Lower-triangular causal mask: (chunk_size, chunk_size)
        causal_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device)
        )

        for i in range(n_chunks):
            q_i = Q_c[:, i]   # (B_batch, chunk_size, d_head)
            k_i = K_c[:, i]
            v_i = V_c[:, i]

            # ---- Intra-chunk causal part ----
            # S_intra: (B_batch, chunk_size, chunk_size)
            # S[b, t, s] = q_i[b,t] · k_i[b,s]  (only s <= t via mask)
            S_intra = torch.bmm(q_i, k_i.transpose(1, 2))   # (B, cs, cs)
            S_intra = S_intra * causal_mask.unsqueeze(0)     # apply causal mask

            # O_intra: (B_batch, chunk_size, d_head)
            O_intra = torch.bmm(S_intra, v_i)

            # ---- Inter-chunk part: use accumulated KV state ----
            # state: (B_batch, d_head, d_head) — represents sum_{past} k^T v
            # O_inter[b, t] = q_i[b, t] @ state[b]  → (B_batch, chunk_size, d_head)
            O_inter = torch.bmm(q_i, state)   # (B_batch, chunk_size, d_head)

            # Combined output for this chunk
            out_i = O_intra + O_inter
            outputs.append(out_i)

            # ---- Update state with this chunk's KV contribution ----
            # state += sum_t  k_i[b,t]^T ⊗ v_i[b,t]
            # = K_i^T @ V_i   shape: (B_batch, d_head, d_head)
            state = state + torch.bmm(k_i.transpose(1, 2), v_i)

        # Reconstruct full output: (B_batch, T_padded, d_head)
        out_padded = torch.cat(outputs, dim=1)

        # Trim padding
        return out_padded[:, :T, :]


# ---------------------------------------------------------------------------
# 2. LightningAttentionLayer — multi-head with QKV + output projections
# ---------------------------------------------------------------------------

class LightningAttentionLayer(nn.Module):
    """Multi-head Lightning Attention-2 layer.

    Projects input to Q, K, V across ``n_heads`` heads, runs per-head
    ``LightningLinearAttn``, then projects back to d_model.

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads. d_model must be divisible by n_heads.
        chunk_size: Tile size passed to each LightningLinearAttn head.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # One LightningLinearAttn per head
        self.heads = nn.ModuleList(
            [LightningLinearAttn(self.d_head, chunk_size) for _ in range(n_heads)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head lightning attention.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V: each (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, T, n_heads, d_head) → list over heads of (B, T, d_head)
        q_heads = q.view(B, T, self.n_heads, self.d_head).unbind(dim=2)
        k_heads = k.view(B, T, self.n_heads, self.d_head).unbind(dim=2)
        v_heads = v.view(B, T, self.n_heads, self.d_head).unbind(dim=2)

        # Per-head attention
        head_outputs = [
            self.heads[h](q_heads[h], k_heads[h], v_heads[h])
            for h in range(self.n_heads)
        ]

        # Concatenate heads: (B, T, d_model)
        out = torch.cat(head_outputs, dim=-1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 3. LightningAttentionBlock — full block with pre-norm + FFN
# ---------------------------------------------------------------------------

class LightningAttentionBlock(nn.Module):
    """Full transformer block: pre-RMSNorm + LightningAttentionLayer + pre-RMSNorm + FFN.

    Follows the pre-norm residual pattern:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads.
        d_ff:       Feed-forward hidden dimension.
        chunk_size: Tile size for tiled linear attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.attn = LightningAttentionLayer(d_model, n_heads, chunk_size)

        self.ffn_norm = _RMSNorm(d_model)
        self.ffn = _FFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        """Apply block: attention residual then FFN residual.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Internal helpers: RMSNorm and FFN (not exported)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (internal helper)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight


class _FFN(nn.Module):
    """Two-layer feed-forward network with GELU activation (internal helper)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


__all__ = [
    "LightningLinearAttn",
    "LightningAttentionLayer",
    "LightningAttentionBlock",
]
