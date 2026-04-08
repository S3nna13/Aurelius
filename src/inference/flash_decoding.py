"""Single-step flash decoding helpers for KV-cache inference."""

from __future__ import annotations

import math

import torch


def flash_decode_step(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    lengths: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one token against a cached key/value history.

    Shapes:
      query: (batch, n_heads, head_dim)
      key_cache: (batch, seq_len, n_heads, head_dim)
      value_cache: (batch, seq_len, n_heads, head_dim)
      lengths: (batch,) valid cache lengths
    """
    if query.dim() != 3 or key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("query must be 3D and caches must be 4D")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must match")
    batch, seq_len, n_heads, head_dim = key_cache.shape
    if query.shape != (batch, n_heads, head_dim):
        raise ValueError(
            f"query shape must be {(batch, n_heads, head_dim)}, got {tuple(query.shape)}"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = torch.einsum("bhd,bshd->bhs", query * scale, key_cache)
    if lengths is not None:
        positions = torch.arange(seq_len, device=key_cache.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshd->bhd", weights, value_cache)


def append_to_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one timestep to existing KV caches."""
    if new_key.dim() != 3 or new_value.dim() != 3:
        raise ValueError("new_key and new_value must be 3D")
    return (
        torch.cat([key_cache, new_key.unsqueeze(1)], dim=1),
        torch.cat([value_cache, new_value.unsqueeze(1)], dim=1),
    )
