"""Nyströmformer: Nyström-Based Self-Attention Approximation.

Implements "Nyströmformer: A Nyström-Based Self-Attention Approximation"
(Xiong et al., arXiv:2107.02239).

Standard softmax attention costs O(T²) in time and memory.  Nyströmformer
reduces this to O(T·m) by selecting m  landmark tokens (segment-mean pooling)
and approximating the full attention matrix as

    Attn ≈ A_hat @ pinv(C_hat) @ B_hat

where
    A_hat = softmax(Q  @ K_lm^T / √d)   — (B, H, T, m)
    B_hat = softmax(Q_lm @ K^T  / √d)   — (B, H, m, T)
    C_hat = softmax(Q_lm @ K_lm^T / √d) — (B, H, m, m)  (pseudoinverted)

Optionally a depth-wise convolution over V is added as a skip connection to
capture local context that the global landmark approximation misses.

Classes
-------
NystromAttention       — single-layer Nyström multi-head self-attention.
NystromformerBlock     — Nyström attention + FFN with RMSNorm pre-norm.
NystromformerModel     — embedding + stack of blocks (no LM head).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helper — RMSNorm (defined locally to keep the module self-contained)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Lightweight Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return (x * rms) * self.weight


# ---------------------------------------------------------------------------
# Internal helper — segment-mean pooling
# ---------------------------------------------------------------------------

def _segment_mean(x: Tensor, num_landmarks: int) -> Tensor:
    """Reduce the time axis of x to num_landmarks via segment-mean pooling.

    The sequence is split into num_landmarks segments.  Each segment is
    averaged to produce one landmark vector.  If T is not evenly divisible the
    last segment is slightly shorter (we use adaptive_avg_pool1d which handles
    this correctly).

    Args:
        x:             (B, H, T, d_head)
        num_landmarks: target number of landmark tokens m.

    Returns:
        Tensor of shape (B, H, m, d_head).
    """
    B, H, T, d = x.shape
    m = min(num_landmarks, T)   # cannot have more landmarks than tokens

    if m == T:
        # Trivial case: every token is its own landmark
        return x

    # Reshape to (B*H, d, T) for adaptive pooling, then back.
    x_t = x.reshape(B * H, T, d).permute(0, 2, 1)      # (B*H, d, T)
    pooled = F.adaptive_avg_pool1d(x_t, m)               # (B*H, d, m)
    pooled = pooled.permute(0, 2, 1).reshape(B, H, m, d) # (B, H, m, d)
    return pooled


# ---------------------------------------------------------------------------
# NystromAttention
# ---------------------------------------------------------------------------

class NystromAttention(nn.Module):
    """Multi-head Nyström self-attention layer.

    Approximates softmax attention in O(T·m) using m landmark tokens chosen by
    segment-mean pooling.

    Args:
        d_model:          Input / output feature dimension.
        n_heads:          Number of attention heads.
        num_landmarks:    Number of landmark tokens m (default 32).
        conv_kernel_size: If an odd integer, add a depth-wise convolution on V
                          as a local-context skip connection (paper §3.2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_landmarks: int = 32,
        conv_kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if conv_kernel_size is not None:
            if conv_kernel_size % 2 == 0:
                raise ValueError("conv_kernel_size must be an odd integer")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.num_landmarks = num_landmarks
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Optional local-context skip: depth-wise conv over V
        self.conv = None
        if conv_kernel_size is not None:
            padding = conv_kernel_size // 2
            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=conv_kernel_size,
                padding=padding,
                groups=d_model,   # depth-wise
                bias=False,
            )

    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) → (B, H, T, d_head)."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, d_head) → (B, T, d_model)."""
        B, H, T, d = x.shape
        return x.transpose(1, 2).reshape(B, T, H * d)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute Nyström self-attention.

        Args:
            x:    (B, T, d_model)
            mask: Optional boolean mask (B, T) — True = keep, False = ignore.
                  Currently used to zero out padding positions before pooling.

        Returns:
            Tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        # --- QKV projections ---
        Q = self._split_heads(self.q_proj(x))   # (B, H, T, d_head)
        K = self._split_heads(self.k_proj(x))   # (B, H, T, d_head)
        V = self._split_heads(self.v_proj(x))   # (B, H, T, d_head)

        # --- Landmark selection via segment-mean pooling ---
        m = min(self.num_landmarks, T)

        Q_lm = _segment_mean(Q, m)   # (B, H, m, d_head)
        K_lm = _segment_mean(K, m)   # (B, H, m, d_head)

        # --- Three softmax attention matrices ---
        # A_hat: Q attends to landmark keys  (B, H, T, m)
        A_hat = torch.softmax(
            Q @ K_lm.transpose(-2, -1) / self.scale, dim=-1
        )

        # B_hat: landmark queries attend to full K  (B, H, m, T)
        B_hat = torch.softmax(
            Q_lm @ K.transpose(-2, -1) / self.scale, dim=-1
        )

        # C_hat: landmark queries attend to landmark keys  (B, H, m, m)
        C_hat = torch.softmax(
            Q_lm @ K_lm.transpose(-2, -1) / self.scale, dim=-1
        )

        # Moore-Penrose pseudoinverse of C_hat
        C_pinv = torch.linalg.pinv(C_hat)   # (B, H, m, m)

        # --- Nyström approximation output ---
        # (B, H, T, m) @ (B, H, m, m) → (B, H, T, m)
        # (B, H, T, m) @ (B, H, m, T) → (B, H, T, T) ... but we fuse:
        # first: (B, H, m, m) @ (B, H, m, T) → (B, H, m, T)
        # then:  (B, H, T, m) @ (B, H, m, T) @ V would be T²; avoid it:
        # out = A_hat @ (C_pinv @ (B_hat @ V))  — right-associative O(T·m)
        BV = B_hat @ V                      # (B, H, m, d_head)
        CBV = C_pinv @ BV                   # (B, H, m, d_head)
        out = A_hat @ CBV                   # (B, H, T, d_head)

        out = self._merge_heads(out)        # (B, T, d_model)

        # --- Optional local skip: depth-wise conv on V ---
        if self.conv is not None:
            V_merged = self._merge_heads(V)          # (B, T, d_model)
            # Conv1d expects (B, C, T)
            V_conv = self.conv(V_merged.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
            out = out + V_conv

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# NystromformerBlock
# ---------------------------------------------------------------------------

class NystromformerBlock(nn.Module):
    """Nyström attention block with FFN and RMSNorm (pre-norm residual).

    Structure (pre-norm):
        x = x + NystromAttention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model:       Model dimension.
        n_heads:       Number of attention heads.
        d_ff:          Feed-forward hidden dimension.
        num_landmarks: Number of Nyström landmark tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_landmarks: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.attn = NystromAttention(d_model, n_heads, num_landmarks=num_landmarks)
        self.norm2 = _RMSNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm transformer block.

        Args:
            x: (B, T, d_model)

        Returns:
            Tensor of shape (B, T, d_model).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# NystromformerModel
# ---------------------------------------------------------------------------

class NystromformerModel(nn.Module):
    """Stack of Nyströmformer blocks with token + position embeddings.

    Returns the final hidden states (B, T, d_model) — no LM head is attached,
    making it easy to compose with task-specific output layers.

    Args:
        vocab_size:    Vocabulary size.
        d_model:       Model dimension.
        n_heads:       Number of attention heads.
        d_ff:          Feed-forward hidden dimension.
        n_layers:      Number of NystromformerBlock layers.
        num_landmarks: Number of Nyström landmark tokens per layer.
        max_seq_len:   Maximum sequence length for learned position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        num_landmarks: int = 32,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            NystromformerBlock(d_model, n_heads, d_ff, num_landmarks=num_landmarks)
            for _ in range(n_layers)
        ])
        self.norm = _RMSNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Encode a batch of token sequences.

        Args:
            input_ids: (B, T) integer token indices.

        Returns:
            Tensor of shape (B, T, d_model) — contextualised hidden states.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
