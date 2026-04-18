"""Sliding-window attention with sink tokens for streaming inference.

Implements the attention-sink pattern from StreamingLLM:
  - keep the first ``S`` tokens as attention sinks
  - keep the most recent ``W`` tokens in a sliding window

Paper notation is used directly in the public API where practical:
  - ``S``: number of sink tokens
  - ``W``: sliding-window size
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SlidingWindowSinkConfig:
    """Configuration for sliding-window sink attention."""

    S: int = 4
    W: int = 256

    def __post_init__(self) -> None:
        if self.S < 0:
            raise ValueError(f"S must be non-negative, got {self.S}")
        if self.W < 1:
            raise ValueError(f"W must be at least 1, got {self.W}")

    @property
    def cache_size(self) -> int:
        return self.S + self.W


def sliding_window_sink_indices(
    T: int, S: int, W: int, device: torch.device | None = None
) -> Tensor:
    """Return the retained KV indices for a length-``T`` prefix."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if S < 0:
        raise ValueError(f"S must be non-negative, got {S}")
    if W < 1:
        raise ValueError(f"W must be at least 1, got {W}")
    if T == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if T <= S + W:
        return torch.arange(T, dtype=torch.long, device=device)

    sink = min(S, T)
    recent_start = max(sink, T - W)
    return torch.cat(
        [
            torch.arange(sink, dtype=torch.long, device=device),
            torch.arange(recent_start, T, dtype=torch.long, device=device),
        ]
    )


def sliding_window_sink_mask(
    q_positions: Tensor,
    k_positions: Tensor,
    S: int,
    W: int,
) -> Tensor:
    """Build the StreamingLLM visibility mask for arbitrary query/key positions.

    Args:
        q_positions: ``(T_q,)`` absolute query positions.
        k_positions: ``(T_k,)`` absolute key positions.
        S: sink-token count.
        W: sliding-window size.

    Returns:
        Bool tensor of shape ``(T_q, T_k)``.
    """
    if q_positions.dim() != 1 or k_positions.dim() != 1:
        raise ValueError("q_positions and k_positions must be 1-D")
    if S < 0:
        raise ValueError(f"S must be non-negative, got {S}")
    if W < 1:
        raise ValueError(f"W must be at least 1, got {W}")

    q = q_positions.view(-1, 1)
    k = k_positions.view(1, -1)
    causal = k <= q
    sink = k < S
    local = k >= (q - W + 1)
    return causal & (sink | local)


def _repeat_kv(x: Tensor, n_heads: int) -> Tensor:
    """Expand grouped KV heads to query-head count."""
    B, n_kv_heads, T, D = x.shape
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    group_size = n_heads // n_kv_heads
    return x.repeat_interleave(group_size, dim=1)


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Numerically stable softmax that returns zeros for fully masked rows."""
    mask_f = mask.to(dtype=scores.dtype)
    floor = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, floor)
    row_max = masked_scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(masked_scores - row_max) * mask_f
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return exp_scores / denom


class SlidingWindowSinkCache:
    """Bounded KV cache that keeps sink tokens and the most recent window."""

    def __init__(self, S: int = 4, W: int = 256) -> None:
        if S < 0:
            raise ValueError(f"S must be non-negative, got {S}")
        if W < 1:
            raise ValueError(f"W must be at least 1, got {W}")
        self.S = S
        self.W = W
        self.reset()

    def reset(self) -> None:
        self.K_cache: Tensor | None = None
        self.V_cache: Tensor | None = None
        self.positions: Tensor | None = None
        self.T_seen = 0

    @property
    def size(self) -> int:
        if self.K_cache is None:
            return 0
        return int(self.K_cache.size(2))

    def update(self, K_new: Tensor, V_new: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Append new KV pairs and return the compressed cache."""
        if K_new.dim() != 4 or V_new.dim() != 4:
            raise ValueError("K_new and V_new must be 4-D (B, H_kv, T, D)")
        if K_new.shape != V_new.shape:
            raise ValueError(
                f"K_new shape {tuple(K_new.shape)} must match V_new shape {tuple(V_new.shape)}"
            )

        T_new = K_new.size(2)
        p_new = torch.arange(
            self.T_seen,
            self.T_seen + T_new,
            dtype=torch.long,
            device=K_new.device,
        )
        self.T_seen += T_new

        if self.K_cache is None:
            K_all = K_new
            V_all = V_new
            p_all = p_new
        else:
            K_all = torch.cat([self.K_cache, K_new], dim=2)
            V_all = torch.cat([self.V_cache, V_new], dim=2)
            p_all = torch.cat([self.positions, p_new], dim=0)

        keep = sliding_window_sink_indices(int(p_all.numel()), self.S, self.W, device=K_all.device)
        self.K_cache = K_all.index_select(2, keep)
        self.V_cache = V_all.index_select(2, keep)
        self.positions = p_all.index_select(0, keep)
        return self.K_cache, self.V_cache, self.positions

    def get(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return cached keys, values, and absolute positions."""
        if self.K_cache is None or self.V_cache is None or self.positions is None:
            raise RuntimeError("Cache is empty; call update() first")
        return self.K_cache, self.V_cache, self.positions


class SlidingWindowSinkAttention(nn.Module):
    """Grouped-query attention with StreamingLLM sink/window masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        S: int = 4,
        W: int = 256,
    ) -> None:
        super().__init__()
        if d_model != n_heads * head_dim:
            raise ValueError(
                f"d_model ({d_model}) must equal n_heads ({n_heads}) * head_dim ({head_dim})"
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

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

        self.cache = SlidingWindowSinkCache(S=S, W=W)

    def reset_cache(self) -> None:
        self.cache.reset()

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        use_cache: bool = False,
        return_attn_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply sliding-window sink attention.

        Args:
            x: ``(B, T, d_model)``.
            key_padding_mask: optional bool tensor ``(B, T)`` with ``True`` for valid tokens.
            use_cache: when ``True``, append this chunk to the internal cache.
            return_attn_weights: when ``True``, also return attention weights.
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            K_cache, V_cache, k_positions = self.cache.update(k, v)
            q_positions = torch.arange(
                self.cache.T_seen - T,
                self.cache.T_seen,
                dtype=torch.long,
                device=x.device,
            )
        else:
            K_cache, V_cache = k, v
            q_positions = torch.arange(T, dtype=torch.long, device=x.device)
            k_positions = q_positions

        K_full = _repeat_kv(K_cache, self.n_heads)
        V_full = _repeat_kv(V_cache, self.n_heads)

        scores = torch.matmul(q, K_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = sliding_window_sink_mask(q_positions, k_positions, self.S, self.W)
        mask = mask.view(1, 1, q_positions.numel(), k_positions.numel()).expand(B, 1, -1, -1)

        query_valid: Tensor | None = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(
                    f"key_padding_mask must have shape {(B, T)}, got {tuple(key_padding_mask.shape)}"
                )
            query_valid = key_padding_mask.to(dtype=torch.bool)
            if use_cache:
                key_valid = torch.ones(B, k_positions.numel(), dtype=torch.bool, device=x.device)
                key_valid[:, -T:] = query_valid
            else:
                key_valid = query_valid
            mask = mask & key_valid[:, None, None, :]

        attn = _masked_softmax(scores.float(), mask).to(dtype=x.dtype)
        out = torch.matmul(attn, V_full)

        if query_valid is not None:
            out = out * query_valid[:, None, :, None].to(dtype=out.dtype)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, q_positions.numel(), self.n_heads * self.head_dim)
        )
        out = self.o_proj(out)

        if return_attn_weights:
            return out, attn
        return out
