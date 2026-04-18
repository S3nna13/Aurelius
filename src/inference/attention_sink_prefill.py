"""Attention-sink prefill for streaming-style causal attention.

This module follows the attention pattern from
"Efficient Streaming Language Models with Attention Sinks": every query can
attend to the first ``S`` sink tokens and the most recent ``W`` tokens within
the causal prefix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_attention_sink_mask(
    seq_len: int,
    S: int,
    W: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the causal attention-sink mask ``M`` of shape ``(T, T)``."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if S < 0:
        raise ValueError("S must be non-negative")
    if W <= 0:
        raise ValueError("W must be positive")

    t = torch.arange(seq_len, device=device)
    i = t.unsqueeze(1)
    j = t.unsqueeze(0)

    sink_band = j < min(S, seq_len)
    window_start = torch.clamp(i - W + 1, min=0)
    window_band = j >= window_start
    causal_band = j <= i
    return causal_band & (sink_band | window_band)


def repeat_kv_heads(X: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Repeat grouped KV heads to match the query head count."""
    if X.dim() != 4:
        raise ValueError("X must have shape (batch, seq_len, n_kv_heads, head_dim)")
    n_kv_heads = X.size(2)
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if n_heads == n_kv_heads:
        return X
    return X.repeat_interleave(n_heads // n_kv_heads, dim=2)


def attention_sink_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    S: int,
    W: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply attention-sink prefill to projected ``Q``, ``K``, and ``V``.

    Args:
        Q: ``(B, T, H, d)`` query states.
        K: ``(B, T, H_kv, d)`` key states.
        V: ``(B, T, H_kv, d)`` value states.
        S: Number of sink tokens.
        W: Sliding-window size.
        attention_mask: Optional ``(B, T)`` mask where non-zero values mark
            valid tokens.

    Returns:
        ``(Y, P)`` where ``Y`` has shape ``(B, T, H, d)`` and ``P`` has shape
        ``(B, H, T, T)``.
    """
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError("Q, K, and V must be 4D tensors")
    if K.shape != V.shape:
        raise ValueError("K and V must have matching shapes")
    if Q.size(0) != K.size(0) or Q.size(1) != K.size(1) or Q.size(3) != K.size(3):
        raise ValueError("Q and K/V batch, sequence, and head_dim must match")

    B, T, H, d = Q.shape
    K = repeat_kv_heads(K, H)
    V = repeat_kv_heads(V, H)

    scale = 1.0 / math.sqrt(d)
    logits = torch.einsum("bthd,bshd->bhts", Q * scale, K)

    M = build_attention_sink_mask(T, S, W, device=Q.device).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        if attention_mask.shape != (B, T):
            raise ValueError("attention_mask must have shape (batch, seq_len)")
        valid_keys = attention_mask.to(dtype=torch.bool, device=Q.device).unsqueeze(1).unsqueeze(1)
        M = M & valid_keys
        valid_queries = attention_mask.to(dtype=Q.dtype, device=Q.device).unsqueeze(1).unsqueeze(-1)
        query_mask = attention_mask.to(dtype=Q.dtype, device=Q.device).unsqueeze(-1).unsqueeze(-1)
    else:
        valid_queries = None
        query_mask = None

    has_support = M.any(dim=-1, keepdim=True)
    masked_logits = logits.masked_fill(~M, torch.finfo(logits.dtype).min)
    max_logits = masked_logits.max(dim=-1, keepdim=True).values
    max_logits = torch.where(has_support, max_logits, torch.zeros_like(max_logits))

    exp_logits = torch.exp(masked_logits - max_logits) * M.to(dtype=logits.dtype)
    denom = exp_logits.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
    P = exp_logits / denom
    if valid_queries is not None:
        P = P * valid_queries

    Y = torch.einsum("bhts,bshd->bthd", P, V)
    if query_mask is not None:
        Y = Y * query_mask
    return Y, P


class AttentionSinkPrefill(nn.Module):
    """Projected attention-sink prefill layer.

    The notation matches the paper: ``S`` is the number of sink tokens and
    ``W`` is the sliding-window length.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        S: int,
        W: int,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0 or n_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("n_heads, n_kv_heads, and head_dim must be positive")
        if d_model != n_heads * head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.S = S
        self.W = W

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def _reshape_q(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        return X.view(B, T, self.n_heads, self.head_dim)

    def _reshape_kv(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        return X.view(B, T, self.n_kv_heads, self.head_dim)

    def forward(
        self,
        X: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if X.dim() != 3 or X.size(-1) != self.d_model:
            raise ValueError("X must have shape (batch, seq_len, d_model)")

        Q = self._reshape_q(self.q_proj(X))
        K = self._reshape_kv(self.k_proj(X))
        V = self._reshape_kv(self.v_proj(X))
        Y, P = attention_sink_prefill(Q, K, V, self.S, self.W, attention_mask=attention_mask)
        B, T, _, _ = Y.shape
        Y = self.o_proj(Y.reshape(B, T, self.n_heads * self.head_dim))
        if attention_mask is not None:
            Y = Y * attention_mask.to(dtype=Y.dtype, device=Y.device).unsqueeze(-1)

        if return_attention:
            return Y, P
        return Y
