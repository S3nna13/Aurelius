"""Aurelius v2 Chunked Prefill Scheduler — efficient prefill for long contexts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkedPrefillScheduler:
    """Schedules prefill in chunks to manage memory and latency.

    Instead of processing the entire prompt at once (O(n^2) memory spike),
    chunks the prompt and processes iteratively.
    """
    chunk_size: int = 2048
    max_concurrent_chunks: int = 16
    deferred_prefill_k: int = 58  # K* parameter

    def split_prompt(self, total_tokens: int) -> list[tuple[int, int]]:
        """Split a prompt into chunks. Returns list of (start, end) token indices."""
        chunks = []
        for start in range(0, total_tokens, self.chunk_size):
            end = min(start + self.chunk_size, total_tokens)
            chunks.append((start, end))
        return chunks

    def estimate_peak_memory_gb(
        self, context_len: int, num_layers: int, num_heads: int, head_dim: int
    ) -> float:
        """Estimate peak memory usage during chunked prefill."""
        # Each chunk: chunk_size * num_layers * num_heads * head_dim * 2 (K+V) * 2 bytes (bf16)
        per_chunk = self.chunk_size * num_layers * num_heads * head_dim * 2 * 2 / (1024**3)
        return per_chunk * min(self.max_concurrent_chunks, len(self.split_prompt(context_len)))

    def estimate_peak_memory_full(
        self, context_len: int, num_layers: int, num_heads: int, head_dim: int
    ) -> float:
        """Estimate peak memory without chunking (for comparison)."""
        full = context_len * num_layers * num_heads * head_dim * 2 * 2 / (1024**3)
        return full

    @property
    def memory_savings_vs_full(self, context_len: int) -> float:
        """Fraction of memory saved by chunked vs full prefill."""
        if context_len <= self.chunk_size:
            return 0.0
        return 1.0 - (self.chunk_size * self.max_concurrent_chunks) / context_len
