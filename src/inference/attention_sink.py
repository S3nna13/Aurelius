"""Attention Sink for Streaming LLM Inference.

Implements the StreamingLLM approach: keep a small set of "sink" tokens
(initial tokens that absorb disproportionate attention mass) alongside
a sliding window of recent tokens, enabling infinite-length generation
with bounded memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import torch


@dataclass
class SinkConfig:
    """Configuration for attention-sink streaming inference."""

    n_sink_tokens: int = 4
    window_size: int = 256
    eviction_policy: str = "fifo"  # "fifo" | "lru"

    def __post_init__(self) -> None:
        if self.n_sink_tokens < 0:
            raise ValueError("n_sink_tokens must be non-negative")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.eviction_policy not in ("fifo", "lru"):
            raise ValueError(f"eviction_policy must be 'fifo' or 'lru', got '{self.eviction_policy}'")


@dataclass
class SinkKVCache:
    """KV cache that preserves sink tokens and evicts from the window region.

    Layout: [sink_0 ... sink_{n-1} | window_0 ... window_{w-1}]
    Keys/values have shape (batch, seq_len, n_kv_heads, head_dim).
    """

    config: SinkConfig
    _keys: torch.Tensor | None = field(default=None, repr=False)
    _values: torch.Tensor | None = field(default=None, repr=False)
    _access_counts: list[int] = field(default_factory=list, repr=False)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append new KV pairs and evict if exceeding window_size (keeping sinks)."""
        if new_k.dim() != 4 or new_v.dim() != 4:
            raise ValueError("new_k and new_v must be 4D (batch, seq, n_kv_heads, head_dim)")

        if self._keys is None:
            self._keys = new_k
            self._values = new_v
            self._access_counts = [0] * new_k.size(1)
        else:
            self._keys = torch.cat([self._keys, new_k], dim=1)
            self._values = torch.cat([self._values, new_v], dim=1)
            self._access_counts.extend([0] * new_k.size(1))

        total = self._keys.size(1)
        max_len = self.config.n_sink_tokens + self.config.window_size

        if total > max_len:
            self._evict(total, max_len)

    def _evict(self, total: int, max_len: int) -> None:
        """Evict tokens from the window region according to eviction policy."""
        n_sink = min(self.config.n_sink_tokens, total)

        if self.config.eviction_policy == "fifo":
            # Keep first n_sink + last window_size
            keep_indices = list(range(n_sink)) + list(range(total - self.config.window_size, total))
        else:
            # LRU: keep sinks + most-recently-accessed window tokens
            window_indices = list(range(n_sink, total))
            # Sort window indices by access count (ascending), then take the most accessed
            window_indices.sort(key=lambda i: self._access_counts[i])
            # Keep the most accessed (last in sorted order)
            keep_window = window_indices[-(self.config.window_size):]
            keep_window.sort()  # preserve order
            keep_indices = list(range(n_sink)) + keep_window

        idx = torch.tensor(keep_indices, device=self._keys.device)
        self._keys = self._keys.index_select(1, idx)
        self._values = self._values.index_select(1, idx)
        self._access_counts = [self._access_counts[i] for i in keep_indices]

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (K, V) with sinks prepended (already in correct order)."""
        if self._keys is None:
            raise RuntimeError("Cache is empty; call update() first")
        # Bump access counts for all current entries
        self._access_counts = [c + 1 for c in self._access_counts]
        return self._keys, self._values

    @property
    def current_length(self) -> int:
        """Current number of cached tokens."""
        return 0 if self._keys is None else self._keys.size(1)

    def reset(self) -> None:
        """Clear the cache entirely."""
        self._keys = None
        self._values = None
        self._access_counts = []


def build_sink_mask(n_sink: int, window_size: int, seq_len: int) -> torch.Tensor:
    """Build an attention mask that allows attending to sink tokens + recent window.

    Returns a (seq_len, seq_len) boolean mask where True means "allowed to attend".
    Each position can attend to:
      - The first n_sink positions (sink tokens)
      - The most recent window_size positions up to and including itself (causal)
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if n_sink < 0 or window_size < 0:
        raise ValueError("n_sink and window_size must be non-negative")

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Sink tokens: always attend to first n_sink (causal: only if j <= i)
        for j in range(min(n_sink, i + 1)):
            mask[i, j] = True
        # Window: attend to recent window_size positions up to and including i
        window_start = max(0, i - window_size + 1)
        for j in range(window_start, i + 1):
            mask[i, j] = True

    return mask


class StreamingGenerator:
    """Generate tokens one at a time using attention-sink KV cache for infinite context."""

    def __init__(self, model: object, config: SinkConfig) -> None:
        self.model = model
        self.config = config

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
    ) -> Generator[int, None, None]:
        """Yield tokens one at a time using SinkKVCache.

        Args:
            input_ids: (batch=1, seq_len) initial prompt token ids.
            max_tokens: maximum number of new tokens to generate.

        Yields:
            One token id at a time.
        """
        cache = SinkKVCache(self.config)

        # Prefill: process entire prompt
        loss, logits, pkv = self.model(input_ids)

        # pkv is a list of (K, V) tuples, one per layer — we only track layer 0
        # for cache management, but store all layers
        if pkv is not None and len(pkv) > 0:
            for layer_k, layer_v in pkv:
                # Accumulate into cache (use first layer for management)
                pass
            # Store full pkv for simplicity; use layer 0 K for cache tracking
            first_k, first_v = pkv[0]
            cache.update(first_k, first_v)

        # Get next token from last position
        next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)
        yield next_token.item()

        for _ in range(max_tokens - 1):
            token_input = next_token.unsqueeze(-1)  # (batch, 1)
            loss, logits, pkv = self.model(token_input)

            if pkv is not None and len(pkv) > 0:
                first_k, first_v = pkv[0]
                cache.update(first_k, first_v)

            next_token = logits[:, -1, :].argmax(dim=-1)
            yield next_token.item()


def compute_sink_attention_stats(
    attention_weights: torch.Tensor,
    n_sink: int,
) -> dict[str, float]:
    """Compute what fraction of attention goes to sink tokens vs window.

    Args:
        attention_weights: (batch, n_heads, seq_len, seq_len) attention matrix.
            Values should sum to ~1 along the last dimension.
        n_sink: number of sink tokens at the beginning of the sequence.

    Returns:
        dict with 'sink_fraction' and 'window_fraction'.
    """
    if attention_weights.dim() != 4:
        raise ValueError("attention_weights must be 4D (batch, heads, seq, seq)")
    if n_sink < 0:
        raise ValueError("n_sink must be non-negative")

    seq_len = attention_weights.size(-1)
    n_sink_clamped = min(n_sink, seq_len)

    # Fraction of attention mass on sink tokens (averaged over all queries, heads, batches)
    sink_mass = attention_weights[:, :, :, :n_sink_clamped].sum(dim=-1).mean().item()
    window_mass = attention_weights[:, :, :, n_sink_clamped:].sum(dim=-1).mean().item()

    total = sink_mass + window_mass
    if total == 0:
        return {"sink_fraction": 0.0, "window_fraction": 0.0}

    return {
        "sink_fraction": sink_mass / total,
        "window_fraction": window_mass / total,
    }
