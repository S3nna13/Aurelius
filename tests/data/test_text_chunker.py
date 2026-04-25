"""Tests for text chunker."""
from __future__ import annotations

import pytest

from src.data.text_chunker import TextChunker


class TestTextChunker:
    def test_empty_text(self):
        tc = TextChunker()
        assert tc.chunk("") == []

    def test_single_chunk(self):
        tc = TextChunker(chunk_size=100, overlap=0)
        chunks = tc.chunk("hello world")
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_multiple_chunks(self):
        tc = TextChunker(chunk_size=5, overlap=1)
        chunks = tc.chunk("abcdefghij")
        assert len(chunks) >= 2
        # each chunk is 5 chars, step is 4
        assert chunks[0] == "abcde"
        assert chunks[1] == "efghi"

    def test_exact_fit(self):
        tc = TextChunker(chunk_size=10, overlap=0)
        chunks = tc.chunk("0123456789")
        assert len(chunks) == 1
        assert chunks[0] == "0123456789"