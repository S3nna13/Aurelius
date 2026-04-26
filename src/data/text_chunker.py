"""Text chunker for splitting documents into fixed-size segments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunker:
    """Split text into fixed-size chunks with optional overlap."""

    chunk_size: int = 512
    overlap: int = 64

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        step = self.chunk_size - self.overlap
        if step <= 0:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += step
        return chunks


TEXT_CHUNKER = TextChunker()
