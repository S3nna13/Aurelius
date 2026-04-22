"""Integration tests for SSE chat-completion stream registration + config."""

from __future__ import annotations

from src.model.config import AureliusConfig
from src.serving import STREAM_HANDLER_REGISTRY, SSEChatStream, parse_sse_event


def test_registered_in_stream_handler_registry() -> None:
    assert "sse_chat" in STREAM_HANDLER_REGISTRY
    assert STREAM_HANDLER_REGISTRY["sse_chat"] is SSEChatStream


def test_config_flag_exists_and_defaults_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "serving_sse_chat_stream_enabled")
    assert cfg.serving_sse_chat_stream_enabled is False


def test_existing_sse_encoder_still_registered() -> None:
    # Additive integration — prior registry entry must remain.
    assert "sse" in STREAM_HANDLER_REGISTRY


def test_end_to_end_stream_produces_parseable_frames() -> None:
    stream = SSEChatStream("aurelius-integration")
    frames = [
        stream.emit_role_delta(),
        stream.emit_content_delta("Hello, "),
        stream.emit_content_delta("world."),
        stream.emit_tool_call_delta(0, name="echo", arguments_chunk='{"x":1}'),
        stream.emit_finish("stop"),
        stream.emit_done(),
    ]
    parsed = [parse_sse_event(f) for f in frames]
    assert parsed[-1] is None
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "Hello, "
    assert parsed[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "echo"
    assert parsed[4]["choices"][0]["finish_reason"] == "stop"
    # Stable stream id across all chunks.
    ids = {p["id"] for p in parsed[:-1]}
    assert len(ids) == 1
