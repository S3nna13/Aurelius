"""Flash-style prefill attention helpers."""

from __future__ import annotations

import math

import torch


def causal_attention_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Compute causal attention scores for a full sequence prefill."""
    if query.shape != key.shape or query.dim() != 4:
        raise ValueError("query and key must be matching 4D tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.einsum("bthd,bshd->bhts", query * scale, key)
    seq_len = query.size(1)
    causal = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1
    )
    return scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))


def flash_prefill(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute full causal attention outputs for prompt prefill."""
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("query, key, and value must match")
    scores = causal_attention_scores(query, key)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", weights, value)


def flash_prefill_with_prefix(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Prefill chunk queries against a longer causal prefix."""
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be 4D")
    if key.shape != value.shape:
        raise ValueError("key and value must match")
    if query.size(0) != key.size(0) or query.size(2) != key.size(2) or query.size(3) != key.size(3):
        raise ValueError("query batch/head dimensions must match key/value")
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.einsum("bthd,bshd->bhts", query * scale, key)
    query_len = query.size(1)
    key_len = key.size(1)
    query_positions = torch.arange(key_len - query_len, key_len, device=query.device)
    key_positions = torch.arange(key_len, device=query.device)
    causal = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", weights, value)


def prefill_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute prefill attention in chunks and concatenate outputs."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    outputs = []
    for start in range(0, query.size(1), chunk_size):
        end = min(start + chunk_size, query.size(1))
        chunk_query = query[:, start:end]
        chunk_out = flash_prefill_with_prefix(chunk_query, key[:, :end], value[:, :end])
        outputs.append(chunk_out)
    return torch.cat(outputs, dim=1)
