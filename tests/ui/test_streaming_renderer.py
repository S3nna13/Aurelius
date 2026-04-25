"""Unit tests for src.ui.streaming_renderer.

16 tests covering all public behaviour of StreamingRenderer.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.streaming_renderer import (
    STREAMING_RENDERER_REGISTRY,
    StreamingRenderer,
    StreamingRendererError,
    StreamingState,
    TokenChunk,
)


# ---------------------------------------------------------------------------
# TokenChunk
# ---------------------------------------------------------------------------


def test_token_chunk_defaults() -> None:
    chunk = TokenChunk(text="hello")
    assert chunk.text == "hello"
    assert chunk.token_id is None
    assert chunk.logprob is None
    assert chunk.finish_reason is None


def test_token_chunk_full_fields() -> None:
    chunk = TokenChunk(text=" world", token_id=42, logprob=-0.5, finish_reason="stop")
    assert chunk.token_id == 42
    assert chunk.logprob == pytest.approx(-0.5)
    assert chunk.finish_reason == "stop"


# ---------------------------------------------------------------------------
# StreamingState
# ---------------------------------------------------------------------------


def test_streaming_state_idle_is_not_streaming() -> None:
    assert StreamingState.STREAMING != StreamingState.IDLE


def test_streaming_state_all_variants_exist() -> None:
    assert StreamingState.IDLE
    assert StreamingState.STREAMING
    assert StreamingState.PAUSED
    assert StreamingState.COMPLETE
    assert StreamingState.ERROR


# ---------------------------------------------------------------------------
# push_chunk / get_text
# ---------------------------------------------------------------------------


def test_push_chunk_appends_to_buffer() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="Hello"))
    assert renderer.get_text() == "Hello"


def test_push_chunk_accumulates_multiple() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="Hello"))
    renderer.push_chunk(TokenChunk(text=", "))
    renderer.push_chunk(TokenChunk(text="world"))
    assert renderer.get_text() == "Hello, world"


def test_push_chunk_transitions_state_to_streaming() -> None:
    renderer = StreamingRenderer()
    assert renderer._state == StreamingState.IDLE
    renderer.push_chunk(TokenChunk(text="x"))
    assert renderer._state == StreamingState.STREAMING


def test_push_chunk_invalid_type_raises() -> None:
    renderer = StreamingRenderer()
    with pytest.raises(StreamingRendererError):
        renderer.push_chunk("not a chunk")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# push_chunks
# ---------------------------------------------------------------------------


def test_push_chunks_batch_appends() -> None:
    renderer = StreamingRenderer()
    chunks = [TokenChunk(text=t) for t in ("The ", "quick ", "brown ", "fox")]
    renderer.push_chunks(chunks)
    assert renderer.get_text() == "The quick brown fox"


def test_push_chunks_invalid_type_raises() -> None:
    renderer = StreamingRenderer()
    with pytest.raises(StreamingRendererError):
        renderer.push_chunks("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# complete / error / reset
# ---------------------------------------------------------------------------


def test_complete_transitions_to_complete() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="done"))
    renderer.complete()
    assert renderer._state == StreamingState.COMPLETE


def test_complete_raises_on_further_push() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="x"))
    renderer.complete()
    with pytest.raises(StreamingRendererError):
        renderer.push_chunk(TokenChunk(text="y"))


def test_error_transitions_to_error() -> None:
    renderer = StreamingRenderer()
    renderer.error("connection lost")
    assert renderer._state == StreamingState.ERROR
    assert renderer._error_message == "connection lost"


def test_error_invalid_message_raises() -> None:
    renderer = StreamingRenderer()
    with pytest.raises(StreamingRendererError):
        renderer.error("")


def test_reset_clears_buffer_and_returns_idle() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="data"))
    renderer.complete()
    renderer.reset()
    assert renderer.get_text() == ""
    assert renderer._state == StreamingState.IDLE
    assert renderer._error_message is None


# ---------------------------------------------------------------------------
# word_count / token_count
# ---------------------------------------------------------------------------


def test_word_count_positive_for_non_empty_buffer() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="The quick brown fox"))
    assert renderer.word_count() > 0


def test_word_count_zero_for_empty_buffer() -> None:
    renderer = StreamingRenderer()
    assert renderer.word_count() == 0


def test_token_count_equals_push_chunk_calls() -> None:
    renderer = StreamingRenderer()
    for i in range(7):
        renderer.push_chunk(TokenChunk(text=f"tok{i}"))
    assert renderer.token_count() == 7


# ---------------------------------------------------------------------------
# render_panel / render_live
# ---------------------------------------------------------------------------


def test_render_panel_does_not_crash() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="Hello Aurelius"))
    console = Console(record=True)
    renderer.render_panel(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_panel_empty_buffer_does_not_crash() -> None:
    renderer = StreamingRenderer()
    console = Console(record=True)
    renderer.render_panel(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_panel_error_state() -> None:
    renderer = StreamingRenderer()
    renderer.error("timeout")
    console = Console(record=True)
    renderer.render_panel(console)
    output = console.export_text()
    assert "ERROR" in output or "timeout" in output


def test_render_live_does_not_crash() -> None:
    renderer = StreamingRenderer()
    renderer.push_chunk(TokenChunk(text="streaming…"))
    console = Console(record=True)
    renderer.render_live(console)
    output = console.export_text()
    assert len(output) > 0


# ---------------------------------------------------------------------------
# Registry / error class
# ---------------------------------------------------------------------------


def test_streaming_renderer_registry_is_dict() -> None:
    assert isinstance(STREAMING_RENDERER_REGISTRY, dict)


def test_streaming_renderer_error_is_exception_subclass() -> None:
    assert issubclass(StreamingRendererError, Exception)
