"""Sink-token attention masking utilities."""

from __future__ import annotations

import torch


def sink_attention_mask(seq_len: int, sink_tokens: int, window_size: int) -> torch.Tensor:
    """Build a causal mask with global sink tokens plus a local trailing window."""
    if seq_len < 0 or sink_tokens < 0 or window_size < 0:
        raise ValueError("seq_len, sink_tokens, and window_size must be non-negative")
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for query_idx in range(seq_len):
        local_start = max(0, query_idx - window_size + 1)
        for key_idx in range(query_idx + 1):
            if key_idx < sink_tokens or key_idx >= local_start:
                mask[query_idx, key_idx] = True
    return mask


def apply_sink_attention(scores: torch.Tensor, sink_tokens: int, window_size: int) -> torch.Tensor:
    """Mask attention scores using the sink-token attention pattern."""
    if scores.dim() != 4:
        raise ValueError("scores must be 4D")
    seq_len = scores.size(-1)
    if scores.size(-2) != seq_len:
        raise ValueError("scores must be square over the last two dims")
    mask = sink_attention_mask(seq_len, sink_tokens, window_size).to(device=scores.device)
    return scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))


def sink_attention_weights(
    scores: torch.Tensor, sink_tokens: int, window_size: int
) -> torch.Tensor:
    """Return normalized sink attention weights."""
    masked = apply_sink_attention(scores, sink_tokens, window_size)
    return torch.softmax(masked, dim=-1)
