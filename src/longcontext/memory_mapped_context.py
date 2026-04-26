"""Memory-mapped context: disk-backed long context storage with random access."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def _new_chunk_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class ContextChunk:
    chunk_id: str = field(default_factory=_new_chunk_id)
    start_pos: int = 0
    end_pos: int = 0
    token_ids: list[int] = field(default_factory=list)
    compressed: bool = False


class MemoryMappedContext:
    """Disk-backed long-context store with chunked random access.

    In this implementation the storage is in-memory (the "memory-mapped"
    label refers to the architecture pattern: fixed-size chunks addressable
    by position range and chunk id).  A real deployment would mmap a file;
    the interface is identical.
    """

    def __init__(self, chunk_size: int = 256, max_chunks: int = 100) -> None:
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self._chunks: dict[str, ContextChunk] = {}  # chunk_id -> chunk
        self._ordered: list[str] = []  # insertion order of chunk_ids
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append_tokens(self, token_ids: list[int]) -> list[ContextChunk]:
        """Split token_ids into chunks of chunk_size, store, and return new chunks."""
        new_chunks: list[ContextChunk] = []
        offset = 0
        while offset < len(token_ids):
            batch = token_ids[offset : offset + self.chunk_size]
            start = self._total_tokens
            end = start + len(batch)
            chunk = ContextChunk(
                start_pos=start,
                end_pos=end,
                token_ids=list(batch),
            )
            self._chunks[chunk.chunk_id] = chunk
            self._ordered.append(chunk.chunk_id)
            self._total_tokens = end
            new_chunks.append(chunk)
            offset += self.chunk_size
        return new_chunks

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> ContextChunk | None:
        """Return chunk by id, or None if not found."""
        return self._chunks.get(chunk_id)

    def get_range(self, start_pos: int, end_pos: int) -> list[int]:
        """Return token_ids covering [start_pos, end_pos) across all chunks."""
        result: list[int] = []
        for cid in self._ordered:
            chunk = self._chunks[cid]
            # Check overlap with [start_pos, end_pos)
            if chunk.end_pos <= start_pos:
                continue
            if chunk.start_pos >= end_pos:
                break
            # Slice within this chunk
            local_start = max(start_pos, chunk.start_pos) - chunk.start_pos
            local_end = min(end_pos, chunk.end_pos) - chunk.start_pos
            result.extend(chunk.token_ids[local_start:local_end])
        return result

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        return self._total_tokens

    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Compression (simulated)
    # ------------------------------------------------------------------

    def compress_chunk(self, chunk_id: str) -> bool:
        """Mark a chunk as compressed (simulated).  Returns True if found."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return False
        chunk.compressed = True
        return True
