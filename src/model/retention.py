"""RetNet: Retentive Network retention mechanism.

Reference: "Retentive Network: A Successor to Transformer for Large Language Models"
           Sun et al., 2023.

Supports three compute modes:
  - Parallel   (training):         O(N^2) — full sequence matrix ops
  - Recurrent  (inference):        O(1) per step — RNN-like state update
  - Chunkwise  (long sequences):   O(C*N) — process in fixed-size chunks
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RetNetConfig:
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.0
    chunk_size: int = 32  # for chunkwise retention


# ---------------------------------------------------------------------------
# Helper: decay gammas
# ---------------------------------------------------------------------------


def build_decay_gammas(n_heads: int) -> Tensor:
    """Compute per-head decay rates γ_h = 1 - 2^(-5 - h*(8/n_heads)).

    Returns shape (n_heads,).
    """
    h = torch.arange(n_heads, dtype=torch.float32)
    gammas = 1.0 - torch.pow(torch.full_like(h, 2.0), -5.0 - h * (8.0 / n_heads))
    return gammas  # (n_heads,)


# ---------------------------------------------------------------------------
# Helper: causal decay mask
# ---------------------------------------------------------------------------


def build_causal_decay_mask(seq_len: int, gamma: float) -> Tensor:
    """Build (T, T) causal retention mask D where D[i,j] = gamma^(i-j) if i>=j else 0."""
    idx = torch.arange(seq_len, dtype=torch.float32)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (T, T), diff[i,j] = i - j
    mask = torch.where(
        diff >= 0,
        torch.pow(torch.full_like(diff, gamma), diff),
        torch.zeros_like(diff),
    )
    return mask  # (T, T)


# ---------------------------------------------------------------------------
# MultiScaleRetention
# ---------------------------------------------------------------------------


class MultiScaleRetention(nn.Module):
    """Multi-scale retention (MSR) — parallel, recurrent, and chunkwise modes.

    For each head h with decay γ_h:
      Retention(X) = (Q K^T ⊙ D) V  (D is causal decay mask)

    Args:
        d_model:  model dimension
        n_heads:  number of retention heads
        dropout:  dropout probability (applied to retention scores)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Gammas: (n_heads,) — registered as buffer so they move with .to(device)
        gammas = build_decay_gammas(n_heads)
        self.register_buffer("gammas", gammas)  # (n_heads,)

        # Projections — one big matrix for all heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # GroupNorm over concatenated head outputs
        self.group_norm = nn.GroupNorm(n_heads, d_model)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) -> (B, n_heads, T, head_dim)."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, n_heads, T, head_dim) -> (B, T, d_model)."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _apply_group_norm(self, x: Tensor) -> Tensor:
        """Apply GroupNorm to (B, T, d_model) tensor."""
        # GroupNorm expects (B, C, *) — treat T as spatial dim
        B, T, C = x.shape
        x = x.transpose(1, 2)   # (B, C, T)
        x = self.group_norm(x)   # (B, C, T)
        return x.transpose(1, 2)  # (B, T, C)

    # ------------------------------------------------------------------
    # Parallel mode (training)
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """Parallel retention. Input (B, T, d_model), output (B, T, d_model)."""
        B, T, _ = x.shape

        Q = self._split_heads(self.W_Q(x))  # (B, H, T, head_dim)
        K = self._split_heads(self.W_K(x))  # (B, H, T, head_dim)
        V = self._split_heads(self.W_V(x))  # (B, H, T, head_dim)

        # Build per-head decay masks: (H, T, T)
        # gammas shape: (H,)
        device = x.device
        decay_masks = torch.stack(
            [build_causal_decay_mask(T, float(g)).to(device) for g in self.gammas],
            dim=0,
        )  # (H, T, T)

        # Retention scores: (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * decay_masks.unsqueeze(0)  # broadcast over batch
        scores = self.drop(scores)

        out = torch.matmul(scores, V)  # (B, H, T, head_dim)
        out = self._merge_heads(out)   # (B, T, d_model)
        out = self._apply_group_norm(out)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Recurrent mode (inference) — single step
    # ------------------------------------------------------------------

    def forward_recurrent(self, x: Tensor, s: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """Recurrent retention for a single token.

        Args:
            x: (B, d_model) — single step input
            s: (B, n_heads, head_dim, head_dim) — recurrent state
            n: current step index (0-based)

        Returns:
            out:   (B, d_model)
            new_s: (B, n_heads, head_dim, head_dim)
        """
        B = x.shape[0]
        H, D = self.n_heads, self.head_dim

        # Project: (B, d_model) -> (B, H, head_dim)
        q = self.W_Q(x).view(B, H, D)
        k = self.W_K(x).view(B, H, D)
        v = self.W_V(x).view(B, H, D)

        # gammas: (H,) -> (1, H, 1, 1) for broadcasting
        gammas = self.gammas.view(1, H, 1, 1)

        # Outer product k^T v: (B, H, head_dim, head_dim)
        # k: (B, H, D) -> (B, H, D, 1); v: (B, H, D) -> (B, H, 1, D)
        kv = torch.einsum("bhd,bhe->bhde", k, v)  # (B, H, D, D)

        # State update: s_n = gamma * s_{n-1} + k_n^T * v_n
        new_s = gammas * s + kv  # (B, H, D, D)

        # Output: q @ s: (B, H, D) @ (B, H, D, D) -> (B, H, D)
        out = torch.einsum("bhd,bhde->bhe", q, new_s) / math.sqrt(D)  # (B, H, D)
        out = out.reshape(B, self.d_model)  # (B, d_model)

        # GroupNorm expects (B, C) — treat as (B, d_model, 1)
        out_t = out.unsqueeze(2)   # (B, d_model, 1) — GroupNorm spatial
        out_t = self.group_norm(out_t)
        out = out_t.squeeze(2)     # (B, d_model)
        out = self.out_proj(out)

        return out, new_s

    # ------------------------------------------------------------------
    # Chunkwise mode (efficient long sequences)
    # ------------------------------------------------------------------

    def forward_chunkwise(self, x: Tensor, chunk_size: int) -> Tensor:
        """Chunkwise retention. Processes in chunks of chunk_size.

        Input (B, T, d_model), output (B, T, d_model).
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        device = x.device

        # gammas: (H,)
        gammas = self.gammas  # (H,)

        outputs: list[Tensor] = []

        # Carry state between chunks: (B, H, D, D)
        state = torch.zeros(B, H, D, D, device=device, dtype=x.dtype)

        # Number of complete chunks; handle remainder
        n_chunks = math.ceil(T / chunk_size)

        token_offset = 0
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, T)
            chunk = x[:, start:end, :]  # (B, C, d_model), C <= chunk_size
            C = chunk.shape[1]

            Q = self._split_heads(self.W_Q(chunk))  # (B, H, C, D)
            K = self._split_heads(self.W_K(chunk))  # (B, H, C, D)
            V = self._split_heads(self.W_V(chunk))  # (B, H, C, D)

            # --- Intra-chunk (parallel within chunk) ---
            # Decay mask for this chunk: (H, C, C)
            intra_decay = torch.stack(
                [build_causal_decay_mask(C, float(g)).to(device) for g in gammas],
                dim=0,
            )  # (H, C, C)

            intra_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
            intra_scores = intra_scores * intra_decay.unsqueeze(0)  # (B, H, C, C)
            intra_out = torch.matmul(intra_scores, V)  # (B, H, C, D)

            # --- Cross-chunk (contribution from carry state) ---
            # For each position i in chunk, the decay from the chunk boundary is
            # gamma^(token_offset + i - (token_offset - 1)) = gamma^(i+1)
            # But more precisely: carry contributes gamma^(i+1) scaled by existing state.
            # cross: Q @ state, scaled per-position by gamma^(position_in_seq - chunk_boundary)
            # Position in sequence: token_offset + i (0-based within chunk: i = 0..C-1)
            # The state was accumulated up to token_offset - 1.
            # Contribution decay: gamma^(position_in_chunk + 1) for position 0..C-1
            pos = torch.arange(1, C + 1, device=device, dtype=x.dtype)  # (C,)
            # gammas: (H,) -> (H, 1)
            decay_cross = torch.pow(
                gammas.unsqueeze(1),  # (H, 1)
                pos.unsqueeze(0),     # (1, C)
            )  # (H, C)

            # state: (B, H, D, D); Q: (B, H, C, D)
            # cross_out[b,h,i] = Q[b,h,i] @ state[b,h] * decay_cross[h,i]
            cross_out = torch.einsum("bhcd,bhde->bhce", Q, state)  # (B, H, C, D)
            cross_out = cross_out * decay_cross.unsqueeze(0).unsqueeze(-1) / math.sqrt(D)
            # (B, H, C, D) * (1, H, C, 1)

            out_chunk = (intra_out + cross_out)  # (B, H, C, D)
            out_chunk = self._merge_heads(out_chunk)  # (B, C, d_model)
            out_chunk = self._apply_group_norm(out_chunk)

            # --- Update carry state ---
            # new_state = gamma^C * state + sum of k_i^T v_i * gamma^(C - 1 - i)
            # Decay the old state by gamma^C
            gamma_C = torch.pow(gammas, float(C))  # (H,)
            state = state * gamma_C.view(1, H, 1, 1)

            # Add new KV contributions: for position i in chunk, decay = gamma^(C-1-i)
            # kv_i = K[i]^T V[i], weighted by gamma^(C-1-i)
            decay_kv = torch.pow(
                gammas.unsqueeze(1),           # (H, 1)
                torch.arange(C - 1, -1, -1, device=device, dtype=x.dtype).unsqueeze(0),  # (1, C)
            )  # (H, C)

            # kv: (B, H, C, D, D) weighted by (B, H, C, 1, 1) -> sum over C
            # Use einsum for efficiency
            # K: (B, H, C, D), V: (B, H, C, D) -> kv: (B, H, C, D, D)
            # weighted sum: (B, H, D, D)
            weighted_kv = torch.einsum(
                "bhci,bhcj,hc->bhij",
                K, V, decay_kv,
            )  # (B, H, D, D)

            state = state + weighted_kv

            token_offset += C
            outputs.append(out_chunk)

        result = torch.cat(outputs, dim=1)  # (B, T, d_model)
        return self.out_proj(result)

    # ------------------------------------------------------------------
    # Default forward — parallel mode
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Default to parallel mode."""
        return self.forward_parallel(x)


# ---------------------------------------------------------------------------
# RetNetBlock
# ---------------------------------------------------------------------------


class RetNetBlock(nn.Module):
    """RetNet block: MSR + FFN with GroupNorm."""

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.retention = MultiScaleRetention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        # SwiGLU-style FFN: two gates
        d_ff = config.d_model * 4
        self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(config.d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.retention(self.norm1(x)))
        # SwiGLU FFN
        h = self.norm2(x)
        h = F.silu(self.ffn_gate(h)) * self.ffn_up(h)
        x = x + self.drop(self.ffn_down(h))
        return x


# ---------------------------------------------------------------------------
# RetNetModel
# ---------------------------------------------------------------------------


class RetNetModel(nn.Module):
    """Stack of RetNet blocks.

    Self-contained — does NOT wrap AureliusTransformer.
    """

    def __init__(self, config: RetNetConfig, n_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Returns logits (B, T, vocab_size)."""
        x = self.embedding(input_ids)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)  # (B, T, vocab_size)
