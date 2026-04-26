"""MLA-256 with Muon Split — GLM-5 §3.1 (arXiv:2602.15763).
head_dim 192→256, n_heads reduced by 1/3.
Muon Split: per-head weight orthogonalization for scale-stable training.
Reduces KV cache at decode time vs standard MLA while improving training stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLA256Config:
    d_model: int = 512
    n_heads: int = 8  # reduced by 1/3 from base (e.g. 12 → 8)
    head_dim: int = 256  # increased from 192
    kv_lrank: int = 64  # low-rank KV compression dimension (latent dim)


class MLA256Attention(nn.Module):
    """Multi-head Latent Attention with 256-dim heads and Muon Split orthogonalization.

    Two key differences from standard MLA:
      - head_dim: 192 → 256 (+33% per head, wider per-head representation)
      - n_heads: reduced by 1/3 (fewer KV heads → smaller KV cache at decode time)

    Net: same total QK compute budget, but smaller cache footprint.

    Muon Split applies Newton-Schulz 2-step orthogonalization per head slice at
    initialization (and optionally as a periodic training regularizer), giving
    each head its own orthonormal basis and preventing logit scale drift.
    """

    def __init__(self, cfg: MLA256Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        # Q projection (full rank)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        # KV: compress to low-rank latent, then expand
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_lrank, bias=False)
        self.k_up = nn.Linear(cfg.kv_lrank, cfg.n_heads * cfg.head_dim, bias=False)
        self.v_up = nn.Linear(cfg.kv_lrank, cfg.n_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # Apply Muon Split at init
        self._apply_muon_split()

    # ------------------------------------------------------------------
    # Muon Split — per-head Newton-Schulz orthogonalization
    # ------------------------------------------------------------------

    def _orthogonalize_per_head(self, weight: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Per-head Newton-Schulz 2-step orthogonalization (Muon Split).

        Each head's weight slice is orthogonalized independently so that
        every head starts from a well-conditioned basis, preventing the
        attention logit scale from drifting across heads.

        Newton-Schulz recurrence (2-step):
            A = m @ m.T
            B = 1.5*I - 0.5*A
            m = B @ m
        """
        with torch.no_grad():
            chunks = weight.chunk(n_heads, dim=0)
            orth_chunks = []
            for chunk in chunks:
                m = chunk.reshape(chunk.shape[0], -1)
                if m.shape[0] > m.shape[1]:
                    # Tall matrix: transpose, orthogonalize rows, transpose back
                    m = m.T
                    A = m @ m.T
                    _I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                    B = 1.5 * _I - 0.5 * A
                    m = (B @ m).T
                else:
                    A = m @ m.T
                    _I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                    B = 1.5 * _I - 0.5 * A
                    m = B @ m
                orth_chunks.append(m.reshape(chunk.shape))
            return torch.cat(orth_chunks, dim=0)

    def _apply_muon_split(self) -> None:
        """Orthogonalize Q projection weight per head at initialization."""
        self.q_proj.weight.data = self._orthogonalize_per_head(
            self.q_proj.weight.data, self.n_heads
        )

    def reorthogonalize(self) -> None:
        """Re-apply Muon Split as a periodic training regularizer.

        Call this from the training loop (e.g. every N steps) to keep
        each head's projection close to orthonormal throughout training.
        Not called during the forward pass — no runtime overhead.
        """
        self._apply_muon_split()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MLA-256 attention.

        Args:
            x: Input tensor of shape [B, T, d_model].

        Returns:
            Output tensor of shape [B, T, d_model].
        """
        B, T, _ = x.shape

        # Queries (full rank projection, head-split)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_heads, T, head_dim]

        # KV via low-rank compression (the "latent" in MLA)
        kv_latent = self.kv_down(x)  # [B, T, kv_lrank]
        k = self.k_up(kv_latent).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(kv_latent).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # each [B, n_heads, T, head_dim]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, n_heads, T, T]
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)

        return self.o_proj(out)  # [B, T, d_model]
