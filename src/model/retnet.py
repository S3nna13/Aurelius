"""RetNet — Retention Networks (Sun et al. 2023).

Reference: "Retentive Network: A Successor to Transformer for Large Language Models"
           Sun et al., 2023.

Key idea: Replace attention with a retention mechanism that supports three compute
modes:
  - Parallel mode  (training):  O(N) via matrix ops with a causal decay mask.
  - Recurrent mode (inference): O(1) per step, like an RNN.

Retention(Q, K, V) = (Q @ K^T ⊙ D) @ V / sqrt(head_dim)
where D[i,j] = gamma^(i-j) if i >= j else 0  (causal decay mask).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Single-head retention
# ---------------------------------------------------------------------------


class SimpleRetention(nn.Module):
    """Single-head retention with a fixed gamma decay factor.

    Parallel mode (training): O(N) compute with matrix ops.
    Recurrent mode (inference): O(1) per step.

    Retention(Q, K, V) = (Q @ K^T ⊙ D) @ V / sqrt(head_dim)
    D[i,j] = gamma^(i-j) if i >= j else 0  (causal decay mask)

    Args:
        d_model:  model (input) hidden dimension.
        head_dim: output dimension for this head.
        gamma:    exponential decay factor in (0, 1).  Default 0.9.
    """

    def __init__(self, d_model: int, head_dim: int, gamma: float = 0.9) -> None:
        super().__init__()
        self.gamma = gamma
        self.head_dim = head_dim
        self.W_Q = nn.Linear(d_model, head_dim, bias=False)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, head_dim, bias=False)

    def _decay_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal decay mask D[i,j] = gamma^(i-j) for i >= j, else 0.

        Shape: (seq_len, seq_len).
        """
        idx = torch.arange(seq_len, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L) diff[i,j] = i - j
        mask = torch.where(
            diff >= 0,
            torch.pow(torch.full_like(diff, self.gamma, dtype=torch.float32), diff.float()),
            torch.zeros(seq_len, seq_len, device=device),
        )
        return mask  # (L, L)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel (training) mode.

        Args:
            x: (B, L, d_model)

        Returns:
            Tensor of shape (B, L, head_dim).
        """
        Q = self.W_Q(x)  # (B, L, head_dim)
        K = self.W_K(x)  # (B, L, head_dim)
        V = self.W_V(x)  # (B, L, head_dim)

        L = x.size(1)
        D = self._decay_mask(L, x.device)  # (L, L)

        # (B, L, L) retention scores with causal decay
        scores = (Q @ K.transpose(-2, -1)) * D.unsqueeze(0)
        scores = scores / math.sqrt(self.head_dim)

        return scores @ V  # (B, L, head_dim)

    def forward_recurrent(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent (inference) mode — process one token at a time.

        State update:
            s_t = gamma * s_{t-1} + K_t^T @ V_t
            y_t = Q_t @ s_t / sqrt(head_dim)

        Args:
            x_t:  (B, 1, d_model) — single new token.
            state: (B, head_dim, head_dim) running KV state, or None for t=0.

        Returns:
            output: (B, 1, head_dim)
            new_state: (B, head_dim, head_dim)
        """
        Q_t = self.W_Q(x_t)  # (B, 1, head_dim)
        K_t = self.W_K(x_t)  # (B, 1, head_dim)
        V_t = self.W_V(x_t)  # (B, 1, head_dim)

        # K_t^T @ V_t: outer product per batch  (B, head_dim, head_dim)
        kv = K_t.transpose(-2, -1) @ V_t

        if state is None:
            new_state = kv
        else:
            new_state = self.gamma * state + kv

        out = (Q_t @ new_state) / math.sqrt(self.head_dim)  # (B, 1, head_dim)
        return out, new_state


# ---------------------------------------------------------------------------
# Multi-head retention
# ---------------------------------------------------------------------------


class MultiScaleRetention(nn.Module):
    """Multi-head retention with per-head gamma schedules.

    Each head i gets a distinct decay:
        gamma_i = 1 - 2^(-5 - floor(8 * i / n_heads))

    This gives gammas ranging from ~0.88 (fast decay) to ~0.997 (slow decay).

    Args:
        d_model: hidden dimension.
        n_heads: number of heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        head_dim = d_model // n_heads

        gammas = [1 - 2 ** (-5 - math.floor(8 * i / n_heads)) for i in range(n_heads)]
        self.heads = nn.ModuleList([SimpleRetention(d_model, head_dim, gamma=g) for g in gammas])
        # GroupNorm over all head outputs (C = d_model, groups = n_heads)
        self.group_norm = nn.GroupNorm(n_heads, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, recurrent: bool = False) -> torch.Tensor:
        """Run all heads then project.

        Args:
            x:         (B, L, d_model)
            recurrent: if True, use recurrent mode; else use parallel mode.

        Returns:
            Tensor of shape (B, L, d_model).
        """
        B, L, _ = x.shape

        if not recurrent:
            head_outs = [h.forward_parallel(x) for h in self.heads]
        else:
            head_outs_list: list[list[torch.Tensor]] = [[] for _ in self.heads]
            states: list[torch.Tensor | None] = [None] * self.n_heads
            for t in range(L):
                x_t = x[:, t : t + 1, :]
                for i, h in enumerate(self.heads):
                    y_t, states[i] = h.forward_recurrent(x_t, states[i])
                    head_outs_list[i].append(y_t)
            head_outs = [torch.cat(steps, dim=1) for steps in head_outs_list]

        combined = torch.cat(head_outs, dim=-1)  # (B, L, d_model)
        combined = combined.transpose(1, 2)  # (B, d_model, L)
        combined = self.group_norm(combined)  # (B, d_model, L)
        combined = combined.transpose(1, 2)  # (B, L, d_model)
        return self.out_proj(combined)


# ---------------------------------------------------------------------------
# RetNet Block
# ---------------------------------------------------------------------------


class RetNetBlock(nn.Module):
    """A single RetNet layer: Multi-Scale Retention + SwiGLU FFN with pre-norm.

    Matches the Aurelius block interface (pre-norm + residual, accepts **kwargs).

    Args:
        config: AureliusConfig (uses d_model, n_heads, rms_norm_eps, d_ff, dropout).
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN
        from .rms_norm import RMSNorm

        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.retention = MultiScaleRetention(config.d_model, config.n_heads)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, recurrent: bool = False, **kwargs) -> torch.Tensor:
        """Pre-norm residual forward pass.

        Args:
            x:         (B, L, d_model)
            recurrent: passed through to MultiScaleRetention.

        Returns:
            Tensor of shape (B, L, d_model).
        """
        x = x + self.retention(self.norm1(x), recurrent=recurrent)
        x = x + self.ffn(self.norm2(x))
        return x
