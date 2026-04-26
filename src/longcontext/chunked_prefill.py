"""Chunked prefill scheduler for memory-bounded long-prompt forward passes.

Reference: Agrawal et al. 2023, "Sarathi: Efficient LLM Inference by Piggybacking
Decodes with Chunked Prefills" (arXiv:2308.16369).

This module is a *pure scheduler*. It does not own model state, caches, or any
attention machinery. Callers supply a ``chunk_fn`` that performs the actual
per-chunk forward pass; the scheduler's job is only to slice the input tensor
into fixed-size (optionally overlapping) windows and glue the per-chunk outputs
back together via ``torch.cat`` along a caller-chosen dimension.

Public surface:
    ChunkedPrefillConfig -- dataclass with chunk_size and overlap
    ChunkedPrefill       -- scheduler with iter_chunks / run_chunk_fn
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch


@dataclass
class ChunkedPrefillConfig:
    """Configuration for :class:`ChunkedPrefill`.

    Attributes:
        chunk_size: Number of tokens per chunk along the sequence dimension.
            Must be strictly positive.
        overlap:    Number of tokens each successive chunk shares with the
            previous one. Must satisfy ``0 <= overlap < chunk_size``.
    """

    chunk_size: int = 512
    overlap: int = 0


class ChunkedPrefill:
    """Pure scheduler that splits a ``[B, S]`` prompt into fixed-size chunks.

    The scheduler performs no model work itself; ``run_chunk_fn`` invokes the
    caller-supplied ``chunk_fn`` once per chunk and concatenates the results.
    """

    def __init__(self, config: ChunkedPrefillConfig) -> None:
        if not isinstance(config, ChunkedPrefillConfig):
            raise TypeError(
                f"config must be a ChunkedPrefillConfig instance, got {type(config).__name__}"
            )
        if config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {config.chunk_size}")
        if config.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {config.overlap}")
        if config.overlap >= config.chunk_size:
            raise ValueError(
                "overlap must be < chunk_size, got "
                f"overlap={config.overlap}, chunk_size={config.chunk_size}"
            )
        self.config = config

    def iter_chunks(self, input_ids: torch.Tensor) -> Iterator[tuple[int, int, torch.Tensor]]:
        """Yield ``(start, end, chunk_ids)`` triples in sequence order.

        The sequence dimension is axis 1. The final chunk may be shorter than
        ``chunk_size`` if ``S`` is not an exact multiple of the effective
        stride. If ``chunk_size >= S`` exactly one chunk covering the full
        sequence is yielded.
        """
        if input_ids.dim() < 2:
            raise ValueError(
                f"input_ids must have at least 2 dims [B, S], got shape {tuple(input_ids.shape)}"
            )
        seq_len = int(input_ids.shape[1])
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        stride = chunk_size - overlap

        if seq_len == 0:
            return

        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            yield start, end, input_ids[:, start:end]
            if end == seq_len:
                break
            start += stride

    def run_chunk_fn(
        self,
        input_ids: torch.Tensor,
        chunk_fn: Callable[[torch.Tensor], torch.Tensor],
        concat_dim: int = 1,
    ) -> torch.Tensor:
        """Run ``chunk_fn`` over every chunk and concat outputs along ``concat_dim``.

        ``chunk_fn`` receives a ``[B, S_chunk]`` (or higher-rank) slice and
        must return a tensor whose shape is compatible with concatenation
        along ``concat_dim``.
        """
        outputs: list[torch.Tensor] = []
        for _, _, chunk in self.iter_chunks(input_ids):
            out = chunk_fn(chunk)
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"chunk_fn must return a torch.Tensor, got {type(out).__name__}")
            outputs.append(out)
        if not outputs:
            # Empty input: return an empty tensor with the input's dtype/device
            # and a zero-length concat axis, preserving the batch dimension.
            shape = list(input_ids.shape)
            if concat_dim < len(shape):
                shape[concat_dim] = 0
            return torch.empty(shape, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat(outputs, dim=concat_dim)


__all__ = ["ChunkedPrefillConfig", "ChunkedPrefill"]
