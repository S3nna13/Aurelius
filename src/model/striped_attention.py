"""Striped Attention — zigzag local/global attention stripes (Brandon et al. 2023).

Alternating full-attention and local-window-attention heads in a zigzag/stripe
pattern achieves good long-range recall with O(n√n) total complexity.

Every other head (by index) uses sliding-window attention; the rest use full
causal attention:
  - head i % 2 == 0  →  global (full causal attention)
  - head i % 2 == 1  →  local  (sliding-window causal attention)

An optional "ratio:N/M" stripe_pattern selects every N-th of M consecutive
heads as global; the remainder are local.

Reference: Brandon et al., "Striped Attention: Faster Ring Attention for Causal
Transformers", 2023 (https://arxiv.org/abs/2311.09431).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StripedAttentionConfig:
    """Configuration for StripedAttention.

    Attributes:
        d_model:        Model dimension (must equal n_heads * head_dim).
        n_heads:        Total number of attention heads.
        head_dim:       Dimension per head. Defaults to d_model // n_heads.
        window_size:    Local window size (in tokens) for local/stripe heads.
        stripe_pattern: "alternating" (default) or "ratio:N/M" — every N-th
                        of M consecutive heads is global.
        dropout:        Attention dropout probability.
    """

    d_model: int = 2048
    n_heads: int = 16
    head_dim: int = 0  # 0 → infer as d_model // n_heads
    window_size: int = 256
    stripe_pattern: str = "alternating"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.head_dim == 0:
            self.head_dim = self.d_model // self.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim ({self.head_dim}) * n_heads ({self.n_heads}) "
                f"must equal d_model ({self.d_model})"
            )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class StripedAttention(nn.Module):
    """Striped (zigzag) attention with alternating global/local heads.

    Each even-indexed head uses standard full causal attention; each odd-indexed
    head uses causal sliding-window (local) attention with a bandwidth of
    ``window_size`` tokens.

    Args:
        cfg: :class:`StripedAttentionConfig` instance.
    """

    def __init__(self, cfg: StripedAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Unified Q/K/V projections (no bias, as is conventional for attention)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.scale = math.sqrt(cfg.head_dim)

        # Pre-compute which heads are global
        self._global_mask: list[bool] = [self._is_global_head(i) for i in range(cfg.n_heads)]

    # ------------------------------------------------------------------
    # Head-type helpers
    # ------------------------------------------------------------------

    def _is_global_head(self, head_idx: int) -> bool:
        """Return True for global (full-causal) heads, False for local heads.

        For "alternating" pattern: even index → global, odd → local.
        For "ratio:N/M": the head is global iff (head_idx % M) < N.

        Args:
            head_idx: Zero-based head index.

        Returns:
            True if the head uses full causal attention.
        """
        pattern = self.cfg.stripe_pattern
        if pattern == "alternating":
            return head_idx % 2 == 0
        if pattern.startswith("ratio:"):
            # e.g. "ratio:1/4"  → every 4 heads, 1 is global (head 0, 4, 8, …)
            ratio_part = pattern[len("ratio:") :]
            n_str, m_str = ratio_part.split("/")
            n, m = int(n_str), int(m_str)
            return (head_idx % m) < n
        raise ValueError(f"Unknown stripe_pattern: {pattern!r}")

    def head_type_mask(self) -> list[str]:
        """Return a list of head type labels for all heads.

        Returns:
            A list such as ``["global", "local", "global", "local", ...]``.
        """
        return ["global" if self._global_mask[i] else "local" for i in range(self.cfg.n_heads)]

    # ------------------------------------------------------------------
    # Attention kernels
    # ------------------------------------------------------------------

    def _full_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Standard scaled dot-product causal attention.

        Args:
            q: Query tensor of shape ``[B, T, head_dim]``.
            k: Key tensor of shape ``[B, T, head_dim]``.
            v: Value tensor of shape ``[B, T, head_dim]``.

        Returns:
            Output tensor of shape ``[B, T, head_dim]``.
        """
        B, T, D = q.shape
        # [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale

        # Causal mask: upper triangle → -inf
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)
        if self.training and self.cfg.dropout > 0.0:
            attn = F.dropout(attn, p=self.cfg.dropout)
        return torch.bmm(attn, v)  # [B, T, head_dim]

    def _local_attention(self, q: Tensor, k: Tensor, v: Tensor, window: int) -> Tensor:
        """Causal sliding-window attention (no flash_attn dependency).

        Position ``i`` can attend to positions ``max(0, i - window + 1) … i``
        (inclusive), i.e. up to ``window`` positions including itself.

        Args:
            q:      Query tensor ``[B, T, head_dim]``.
            k:      Key tensor ``[B, T, head_dim]``.
            v:      Value tensor ``[B, T, head_dim]``.
            window: Sliding-window size in tokens.

        Returns:
            Output tensor of shape ``[B, T, head_dim]``.
        """
        B, T, D = q.shape
        # Full score matrix [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale

        # Build combined causal + window mask [T, T]
        idx_i = torch.arange(T, device=q.device).unsqueeze(1)  # [T, 1]
        idx_j = torch.arange(T, device=q.device).unsqueeze(0)  # [1, T]

        # Allowed: j <= i  AND  j >= i - window + 1
        allowed = (idx_j <= idx_i) & (idx_j >= idx_i - window + 1)
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype)
        mask[~allowed] = float("-inf")

        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        if self.training and self.cfg.dropout > 0.0:
            attn = F.dropout(attn, p=self.cfg.dropout)
        return torch.bmm(attn, v)  # [B, T, head_dim]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Compute striped attention over the input sequence.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            Output tensor of shape ``[B, T, d_model]``.
        """
        B, T, _ = x.shape
        H = self.cfg.n_heads
        D = self.cfg.head_dim

        # Project to Q, K, V then split into per-head views
        # [B, T, H*D]  →  [B, T, H, D]  →  [H, B, T, D]
        Q = self.q_proj(x).view(B, T, H, D).permute(2, 0, 1, 3)
        K = self.k_proj(x).view(B, T, H, D).permute(2, 0, 1, 3)
        V = self.v_proj(x).view(B, T, H, D).permute(2, 0, 1, 3)

        head_outputs: list[Tensor] = []
        for i in range(H):
            q_i, k_i, v_i = Q[i], K[i], V[i]  # each [B, T, D]
            if self._global_mask[i]:
                out_i = self._full_attention(q_i, k_i, v_i)
            else:
                out_i = self._local_attention(q_i, k_i, v_i, self.cfg.window_size)
            head_outputs.append(out_i)  # [B, T, D]

        # Concatenate along head dimension → [B, T, H*D]
        concat = torch.cat(head_outputs, dim=-1)
        return self.out_proj(concat)  # [B, T, d_model]
