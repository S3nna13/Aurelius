"""Chunked prefill scheduler for Aurelius long-context inference.

Splits long token sequences into overlapping chunks for incremental KV-cache
construction, then provides helpers to estimate memory usage and merge caches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkPrefillConfig:
    chunk_size: int = 512
    overlap: int = 64
    max_chunks: int = 32


@dataclass
class ChunkResult:
    chunk_idx: int
    token_ids: list[int]
    kv_cache: dict | None
    is_last: bool


class ChunkPrefillScheduler:
    """Splits token sequences and manages chunked prefill scheduling."""

    # ------------------------------------------------------------------
    # split
    # ------------------------------------------------------------------

    def split(
        self,
        token_ids: list[int],
        config: ChunkPrefillConfig | None = None,
    ) -> list[ChunkResult]:
        """Split *token_ids* into overlapping chunks.

        Each chunk except the last has length ``config.chunk_size``.
        Consecutive chunks share ``config.overlap`` tokens at their
        boundary.  At most ``config.max_chunks`` chunks are produced;
        the last chunk beyond that limit is clipped to the first
        ``config.max_chunks`` chunks.
        """
        if config is None:
            config = ChunkPrefillConfig()

        stride = config.chunk_size - config.overlap
        if stride <= 0:
            raise ValueError(
                f"chunk_size ({config.chunk_size}) must be greater than overlap ({config.overlap})"
            )

        chunks: list[ChunkResult] = []
        start = 0
        idx = 0
        total = len(token_ids)

        while start < total and idx < config.max_chunks:
            end = min(start + config.chunk_size, total)
            is_last = end >= total or (idx + 1) >= config.max_chunks
            chunks.append(
                ChunkResult(
                    chunk_idx=idx,
                    token_ids=token_ids[start:end],
                    kv_cache=None,
                    is_last=is_last,
                )
            )
            if is_last:
                break
            start += stride
            idx += 1

        return chunks

    # ------------------------------------------------------------------
    # memory estimation
    # ------------------------------------------------------------------

    def estimate_memory(
        self,
        token_ids: list[int],
        n_layers: int,
        n_heads: int,
        head_dim: int,
        config: ChunkPrefillConfig | None = None,
    ) -> dict:
        """Estimate KV-cache memory for chunked prefill.

        Returns
        -------
        dict with keys:
            ``n_chunks``        – number of chunks produced
            ``peak_kv_bytes``   – bytes required for one chunk's KV cache
            ``total_kv_bytes``  – bytes for all chunks' KV caches
        """
        if config is None:
            config = ChunkPrefillConfig()

        chunks = self.split(token_ids, config)
        n_chunks = len(chunks)

        # kv_bytes per token per layer: 2 (K and V) * n_heads * head_dim * 2 (float16)
        bytes_per_token_per_layer = 2 * n_heads * head_dim * 2

        peak_tokens = max((len(c.token_ids) for c in chunks), default=0)
        peak_kv_bytes = peak_tokens * n_layers * bytes_per_token_per_layer
        total_kv_bytes = (
            sum(len(c.token_ids) for c in chunks) * n_layers * bytes_per_token_per_layer
        )

        return {
            "n_chunks": n_chunks,
            "peak_kv_bytes": peak_kv_bytes,
            "total_kv_bytes": total_kv_bytes,
        }

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------

    def merge_kv_caches(self, results: list[ChunkResult]) -> list[dict]:
        """Naive concatenation of kv_cache entries from all results.

        Returns a list of dicts (one per chunk), filtering out chunks whose
        kv_cache is None.
        """
        return [r.kv_cache for r in results if r.kv_cache is not None]
