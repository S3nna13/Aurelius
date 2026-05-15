"""Aurelius v2 Attention Efficiency — cross-layer KV sharing, attention sinks, dynamic sparse attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrossLayerKVSharing:
    """Shares KV between adjacent transformer layers to reduce memory.

    Inspired by DuoAttention and layer-sharing techniques.
    Selected layers share KV representations, reducing total KV cache size.
    """
    sharing_pattern: str = "alternate"  # alternate, pairs, sparse
    layers_per_group: int = 2

    def should_share(self, layer_idx: int) -> bool:
        if self.sharing_pattern == "alternate":
            return layer_idx % 2 == 1
        elif self.sharing_pattern == "pairs":
            return layer_idx % self.layers_per_group != 0
        return False

    @property
    def memory_savings_pct(self) -> float:
        if self.sharing_pattern == "alternate":
            return 50.0
        elif self.sharing_pattern == "pairs":
            return 100.0 / self.layers_per_group * (self.layers_per_group - 1) / self.layers_per_group
        return 0.0


@dataclass
class AttentionSinkManager:
    """Manages attention sink tokens for long-context stability.

    Keeps a fixed set of initial tokens always attended to,
    preventing information loss in very long contexts.
    """
    num_sink_tokens: int = 4
    sink_token_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def apply_sinks(self, attention_scores: Any, context_len: int) -> Any:
        """Ensure sink tokens always receive attention."""
        # In production: modify attention scores so sink tokens
        # maintain minimum attention weight regardless of position
        return attention_scores

    def sink_memory_gb(self, head_dim: int, num_heads: int, num_layers: int) -> float:
        """Memory used by sink tokens."""
        per_token = head_dim * 2 * num_heads * num_layers * 2 / (1024**3)  # K and V, 2 bytes (bf16)
        return per_token * self.num_sink_tokens


@dataclass
class DynamicSparseAttention:
    """Dynamic sparse attention for long-context efficiency.

    Selects which tokens to attend to based on relevance scores,
    reducing O(n^2) to O(n*k) where k << n.
    """
    sparsity_ratio: float = 0.25  # attend to 25% of tokens
    top_k_tokens: int = 256
    sliding_window: int = 128

    def compute_attention_mask(self, context_len: int, query_positions: list[int]) -> dict[int, list[int]]:
        """Compute sparse attention mask for given query positions.

        Each query attends to:
        - Sliding window around it
        - Top-k most relevant tokens
        """
        mask = {}
        for qpos in query_positions:
            # Local sliding window
            start = max(0, qpos - self.sliding_window)
            end = min(context_len, qpos + self.sliding_window)
            attended = list(range(start, end))
            # Add top-k most relevant (placeholder)
            for i in range(min(self.top_k_tokens, context_len)):
                if i not in attended:
                    attended.append(i)
            mask[qpos] = attended[:self.top_k_tokens + self.sliding_window * 2]
        return mask
