"""Tests for src/serving/response_streamer.py — ≥28 test cases."""

from __future__ import annotations

import pytest

from src.serving.response_streamer import (
    RESPONSE_STREAMER_REGISTRY,
    ResponseStreamer,
    StreamChunk,
    StreamConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _streamer(chunk_size: int = 4, buffer_size: int = 64) -> ResponseStreamer:
    return ResponseStreamer(StreamConfig(chunk_size=chunk_size, buffer_size=buffer_size))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in RESPONSE_STREAMER_REGISTRY

    def test_registry_default_is_class(self):
        assert RESPONSE_STREAMER_REGISTRY["default"] is ResponseStreamer


# ---------------------------------------------------------------------------
# Frozen dataclass contracts
# ---------------------------------------------------------------------------


class TestStreamChunkFrozen:
    def test_chunk_fields(self):
        c = StreamChunk(chunk_id=0, text="hello", is_final=True)
        assert c.chunk_id == 0
        assert c.text == "hello"
        assert c.is_final is True

    def test_chunk_default_not_final(self):
        c = StreamChunk(chunk_id=0, text="x")
        assert c.is_final is False

    def test_chunk_default_metadata(self):
        c = StreamChunk(chunk_id=0, text="x")
        assert c.metadata == {}

    def test_chunk_is_frozen(self):
        c = StreamChunk(chunk_id=0, text="abc")
        with pytest.raises((AttributeError, TypeError)):
            c.text = "changed"  # type: ignore[misc]

    def test_chunk_metadata_custom(self):
        c = StreamChunk(chunk_id=1, text="t", metadata={"key": "val"})
        assert c.metadata["key"] == "val"


class TestStreamConfigFrozen:
    def test_config_defaults(self):
        cfg = StreamConfig()
        assert cfg.chunk_size == 4
        assert cfg.buffer_size == 64

    def test_config_custom(self):
        cfg = StreamConfig(chunk_size=8, buffer_size=16)
        assert cfg.chunk_size == 8
        assert cfg.buffer_size == 16

    def test_config_is_frozen(self):
        cfg = StreamConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.chunk_size = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ResponseStreamer default config
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_config_applied(self):
        s = ResponseStreamer()
        assert s.config.chunk_size == 4
        assert s.config.buffer_size == 64


# ---------------------------------------------------------------------------
# stream() — character-based chunking
# ---------------------------------------------------------------------------


class TestStream:
    def test_empty_text_single_final_chunk(self):
        s = _streamer()
        chunks = s.stream("")
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].is_final is True
        assert chunks[0].chunk_id == 0

    def test_chunk_id_increments_from_zero(self):
        s = _streamer(chunk_size=2)
        chunks = s.stream("abcdef")
        assert [c.chunk_id for c in chunks] == [0, 1, 2]

    def test_last_chunk_is_final(self):
        s = _streamer(chunk_size=3)
        chunks = s.stream("hello")
        assert chunks[-1].is_final is True

    def test_only_last_chunk_is_final(self):
        s = _streamer(chunk_size=2)
        chunks = s.stream("abcdef")
        for c in chunks[:-1]:
            assert c.is_final is False

    def test_eight_char_chunk_size_4_gives_two_chunks(self):
        s = _streamer(chunk_size=4)
        chunks = s.stream("12345678")
        assert len(chunks) == 2
        assert chunks[0].text == "1234"
        assert chunks[1].text == "5678"

    def test_text_shorter_than_chunk_size(self):
        s = _streamer(chunk_size=10)
        chunks = s.stream("hi")
        assert len(chunks) == 1
        assert chunks[0].text == "hi"
        assert chunks[0].is_final is True

    def test_exact_multiple_chunks(self):
        s = _streamer(chunk_size=3)
        chunks = s.stream("abcdef")
        assert len(chunks) == 2
        assert chunks[0].text == "abc"
        assert chunks[1].text == "def"

    def test_non_multiple_remainder(self):
        s = _streamer(chunk_size=4)
        chunks = s.stream("abcde")
        assert len(chunks) == 2
        assert chunks[1].text == "e"


# ---------------------------------------------------------------------------
# stream_tokens() — per-token chunking
# ---------------------------------------------------------------------------


class TestStreamTokens:
    def test_each_token_is_one_chunk(self):
        s = _streamer()
        tokens = ["hello", " ", "world"]
        chunks = s.stream_tokens(tokens)
        assert len(chunks) == 3

    def test_chunk_ids_sequential(self):
        s = _streamer()
        chunks = s.stream_tokens(["a", "b", "c"])
        assert [c.chunk_id for c in chunks] == [0, 1, 2]

    def test_last_token_is_final(self):
        s = _streamer()
        chunks = s.stream_tokens(["x", "y"])
        assert chunks[-1].is_final is True
        assert chunks[0].is_final is False

    def test_single_token(self):
        s = _streamer()
        chunks = s.stream_tokens(["only"])
        assert len(chunks) == 1
        assert chunks[0].is_final is True

    def test_empty_tokens_single_final_chunk(self):
        s = _streamer()
        chunks = s.stream_tokens([])
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].text == ""

    def test_token_texts_preserved(self):
        s = _streamer()
        tokens = ["The", " cat", " sat"]
        chunks = s.stream_tokens(tokens)
        assert [c.text for c in chunks] == tokens


# ---------------------------------------------------------------------------
# collect()
# ---------------------------------------------------------------------------


class TestCollect:
    def test_collect_rejoins_stream_chunks(self):
        s = _streamer(chunk_size=3)
        original = "hello world"
        chunks = s.stream(original)
        assert s.collect(chunks) == original

    def test_collect_rejoins_token_chunks(self):
        s = _streamer()
        tokens = ["The", " fox", " jumps"]
        chunks = s.stream_tokens(tokens)
        assert s.collect(chunks) == "The fox jumps"

    def test_collect_empty_list(self):
        s = _streamer()
        assert s.collect([]) == ""

    def test_collect_single_chunk(self):
        s = _streamer()
        chunks = [StreamChunk(chunk_id=0, text="alone", is_final=True)]
        assert s.collect(chunks) == "alone"


# ---------------------------------------------------------------------------
# buffer_chunks()
# ---------------------------------------------------------------------------


class TestBufferChunks:
    def test_empty_input_returns_one_empty_page(self):
        s = _streamer()
        result = s.buffer_chunks([])
        assert result == [[]]

    def test_chunks_fit_in_one_page(self):
        s = _streamer(chunk_size=1, buffer_size=64)
        chunks = s.stream("hello")
        pages = s.buffer_chunks(chunks)
        assert len(pages) == 1
        assert len(pages[0]) == 5

    def test_chunks_split_into_multiple_pages(self):
        s = _streamer(chunk_size=1, buffer_size=3)
        chunks = s.stream("abcdef")  # 6 chunks
        pages = s.buffer_chunks(chunks)
        assert len(pages) == 2
        assert len(pages[0]) == 3
        assert len(pages[1]) == 3

    def test_buffer_remainder_page(self):
        s = _streamer(chunk_size=1, buffer_size=4)
        chunks = s.stream("abcde")  # 5 chunks
        pages = s.buffer_chunks(chunks)
        assert len(pages) == 2
        assert len(pages[0]) == 4
        assert len(pages[1]) == 1

    def test_buffer_chunks_all_chunks_preserved(self):
        s = _streamer(chunk_size=2, buffer_size=3)
        original = "abcdefgh"
        chunks = s.stream(original)
        pages = s.buffer_chunks(chunks)
        all_chunks = [c for page in pages for c in page]
        assert s.collect(all_chunks) == original
