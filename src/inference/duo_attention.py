"""DuoAttention — retrieval vs. streaming head classification and masking.

DuoAttention (Xiao et al., 2024) classifies attention heads into:
* **Retrieval heads** — attend broadly across the full KV cache.
* **Streaming heads** — attend locally, keeping only sink tokens + recent window.

This module provides masks and application logic; head classification is
performed offline by ``scripts/classify_attention_heads.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DuoAttentionConfig:
    """Configuration for DuoAttention head masks."""

    retrieval_heads: dict[int, list[int]]
    """Mapping ``{layer_idx: [head_indices]}`` for full-KV retrieval heads."""

    streaming_heads: dict[int, list[int]]
    """Mapping ``{layer_idx: [head_indices]}`` for streaming heads."""

    sink_size: int = 4
    """Number of initial sink tokens to retain for streaming heads."""

    recent_size: int = 512
    """Number of trailing recent tokens to retain for streaming heads."""


class DuoAttentionManager:
    """Apply DuoAttention KV masks to attention weights."""

    def __init__(self, config: DuoAttentionConfig) -> None:
        self.config = config

    def get_kv_mask(self, layer_idx: int, head_idx: int, seq_len: int) -> Tensor:
        """Return a boolean mask over KV positions for the given layer/head.

        Retrieval heads receive an all-``True`` mask (full KV).  Streaming heads
        mask out everything except the first ``sink_size`` and last ``recent_size``
        positions when the sequence is long enough.

        Args:
            layer_idx: Transformer layer index.
            head_idx: Attention head index within the layer.
            seq_len: Length of the KV sequence to mask.

        Returns:
            Boolean tensor of shape ``(seq_len,)`` where ``True`` means the
            position is visible to the attention head.
        """
        mask = torch.ones(seq_len, dtype=torch.bool)
        streaming_heads = self.config.streaming_heads.get(layer_idx, [])
        if head_idx in streaming_heads:
            if seq_len > self.config.sink_size + self.config.recent_size:
                mask[self.config.sink_size : -self.config.recent_size] = False
        return mask

    def apply(self, attn_weights: Tensor, layer_idx: int, head_idx: int) -> Tensor:
        """Zero out masked KV positions in *attn_weights* before softmax.

        Args:
            attn_weights: Pre-softmax attention scores, typically
                ``(..., seq_len)`` or ``(..., query_len, key_len)``.
            layer_idx: Transformer layer index.
            head_idx: Attention head index.

        Returns:
            Masked attention weights where invisible positions are set to
            ``float('-inf')``.
        """
        seq_len = attn_weights.shape[-1]
        mask = self.get_kv_mask(layer_idx, head_idx, seq_len)
        mask = mask.to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        return attn_weights

    def apply_to_scores(
        self,
        scores: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """Apply masks to a full multi-head score tensor.

        Convenience helper when *scores* has shape
        ``(batch, n_heads, query_len, key_len)``.

        Args:
            scores: Pre-softmax scores for all heads in a layer.
            layer_idx: Transformer layer index.

        Returns:
            Masked scores with the same shape as *scores*.
        """
        batch, n_heads, q_len, k_len = scores.shape
        for h in range(n_heads):
            mask = self.get_kv_mask(layer_idx, h, k_len).to(scores.device)
            # Expand mask to (1, 1, 1, k_len) for broadcasting
            scores[:, h, :, :] = scores[:, h, :, :].masked_fill(
                ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        return scores
