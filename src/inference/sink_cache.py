"""Sink-token cache utilities for sliding-window decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def sink_window_indices(total_tokens: int, sink_tokens: int, window_size: int) -> list[int]:
    """Keep fixed sink tokens plus the most recent decode window."""
    if total_tokens < 0 or sink_tokens < 0 or window_size < 0:
        raise ValueError("total_tokens, sink_tokens, and window_size must be non-negative")
    sink = list(range(min(sink_tokens, total_tokens)))
    trailing_start = max(sink_tokens, total_tokens - window_size)
    trailing = list(range(trailing_start, total_tokens))
    return sink + [index for index in trailing if index not in sink]


def compress_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sink_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Select the sink + trailing window from KV caches."""
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must match")
    if key_cache.dim() != 4:
        raise ValueError("key_cache and value_cache must be 4D")
    indices = sink_window_indices(key_cache.size(1), sink_tokens, window_size)
    index_tensor = torch.tensor(indices, device=key_cache.device)
    return (
        key_cache.index_select(1, index_tensor),
        value_cache.index_select(1, index_tensor),
        indices,
    )


@dataclass
class SinkCache:
    sink_tokens: int
    window_size: int
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one timestep and compress to the sink-window view."""
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError("key and value must be 3D")
        if self.key_cache is None:
            self.key_cache = key.unsqueeze(1)
            self.value_cache = value.unsqueeze(1)
        else:
            self.key_cache = torch.cat([self.key_cache, key.unsqueeze(1)], dim=1)
            self.value_cache = torch.cat([self.value_cache, value.unsqueeze(1)], dim=1)
        self.key_cache, self.value_cache, _ = compress_kv_cache(
            self.key_cache,
            self.value_cache,
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
        )

    def current_length(self) -> int:
        return 0 if self.key_cache is None else self.key_cache.size(1)
