"""Streams response tokens with backpressure control.

Generated text is split into fixed-size character chunks or per-token chunks.
Chunks are wrapped in :class:`StreamChunk` dataclasses and can optionally be
grouped into buffer-sized pages to simulate backpressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamChunk:
    """A single streaming unit of response text.

    Args:
        chunk_id: Zero-based position of this chunk in the stream.
        text:     The text content of this chunk.
        is_final: ``True`` for the last chunk in a stream.
        metadata: Optional key-value annotations.
    """

    chunk_id: int
    text: str
    is_final: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for a :class:`ResponseStreamer`.

    Args:
        chunk_size:  Number of characters per chunk when streaming raw text
                     (default 4).
        buffer_size: Maximum number of chunks in a single backpressure buffer
                     group (default 64).
    """

    chunk_size: int = 4
    buffer_size: int = 64


class ResponseStreamer:
    """Converts text or token sequences into :class:`StreamChunk` lists.

    Args:
        config: A :class:`StreamConfig` instance.  Defaults to
                ``StreamConfig()`` when ``None``.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config: StreamConfig = config if config is not None else StreamConfig()

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def stream(self, text: str) -> list[StreamChunk]:
        """Split *text* into character-based chunks.

        Each chunk contains at most ``config.chunk_size`` characters.
        The final chunk always has ``is_final=True``.

        Empty *text* produces a single empty final chunk.

        Args:
            text: The full response text to stream.

        Returns:
            Ordered list of :class:`StreamChunk` objects.
        """
        if not text:
            return [StreamChunk(chunk_id=0, text="", is_final=True)]

        size = self.config.chunk_size
        chunks: list[StreamChunk] = []
        total = math.ceil(len(text) / size)

        for i in range(total):
            start = i * size
            end = start + size
            is_final = i == total - 1
            chunks.append(StreamChunk(chunk_id=i, text=text[start:end], is_final=is_final))

        return chunks

    def stream_tokens(self, tokens: list[str]) -> list[StreamChunk]:
        """Wrap each token as a separate :class:`StreamChunk`.

        The last token receives ``is_final=True``.  An empty *tokens* list
        returns a single empty final chunk.

        Args:
            tokens: Pre-tokenized strings.

        Returns:
            Ordered list of :class:`StreamChunk` objects.
        """
        if not tokens:
            return [StreamChunk(chunk_id=0, text="", is_final=True)]

        chunks: list[StreamChunk] = []
        last_idx = len(tokens) - 1
        for i, token in enumerate(tokens):
            chunks.append(StreamChunk(chunk_id=i, text=token, is_final=(i == last_idx)))

        return chunks

    # ------------------------------------------------------------------
    # Collection & buffering
    # ------------------------------------------------------------------

    def collect(self, chunks: list[StreamChunk]) -> str:
        """Reassemble streamed chunks back into a single string.

        Args:
            chunks: Any ordered sequence of :class:`StreamChunk` objects.

        Returns:
            The concatenated text of all chunks.
        """
        return "".join(c.text for c in chunks)

    def buffer_chunks(self, chunks: list[StreamChunk]) -> list[list[StreamChunk]]:
        """Group *chunks* into pages of at most ``config.buffer_size`` items.

        This simulates backpressure by batching the output into discrete
        pages that a consumer can process before requesting the next page.

        Args:
            chunks: A flat list of :class:`StreamChunk` objects.

        Returns:
            A list of pages; each page is a list of :class:`StreamChunk`.
            Returns ``[[]]`` for an empty input.
        """
        if not chunks:
            return [[]]

        size = self.config.buffer_size
        pages: list[list[StreamChunk]] = []
        for start in range(0, len(chunks), size):
            pages.append(chunks[start : start + size])

        return pages


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RESPONSE_STREAMER_REGISTRY: dict[str, type] = {
    "default": ResponseStreamer,
}
