"""Tests for OpenAI-compatible SSE chat-completion streaming."""

from __future__ import annotations

import json

import pytest

from src.serving.sse_chat_stream import (
    ChatCompletionChunk,
    ChoiceDelta,
    SSEChatStream,
    SSEParseError,
    parse_sse_event,
)


def _payload(frame: str) -> dict:
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    return json.loads(frame[len("data: ") : -2])


def test_role_delta_contains_assistant_role() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_role_delta()
    obj = _payload(frame)
    assert obj["choices"][0]["delta"]["role"] == "assistant"
    assert obj["choices"][0]["finish_reason"] is None


def test_content_delta_carries_content_verbatim() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_content_delta("hello world")
    obj = _payload(frame)
    assert obj["choices"][0]["delta"]["content"] == "hello world"
    assert obj["object"] == "chat.completion.chunk"


def test_tool_call_delta_has_name_and_arguments() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_tool_call_delta(0, name="search", arguments_chunk='{"q":')
    obj = _payload(frame)
    tc = obj["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == '{"q":'


def test_tool_call_delta_args_only() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_tool_call_delta(0, name=None, arguments_chunk='"hi"}')
    obj = _payload(frame)
    tc = obj["choices"][0]["delta"]["tool_calls"][0]
    assert "name" not in tc["function"]
    assert tc["function"]["arguments"] == '"hi"}'


def test_tool_call_index_must_be_int() -> None:
    stream = SSEChatStream("aurelius-test")
    with pytest.raises(ValueError):
        stream.emit_tool_call_delta(-1, name="x", arguments_chunk=None)
    with pytest.raises(ValueError):
        stream.emit_tool_call_delta("0", name="x", arguments_chunk=None)  # type: ignore[arg-type]


def test_finish_reason_present_in_last_chunk() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_finish("stop")
    obj = _payload(frame)
    assert obj["choices"][0]["finish_reason"] == "stop"
    assert obj["choices"][0]["delta"] == {}


def test_emit_done_exact_format() -> None:
    stream = SSEChatStream("aurelius-test")
    assert stream.emit_done() == "data: [DONE]\n\n"


def test_parse_round_trip() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_content_delta("round-trip")
    obj = parse_sse_event(frame)
    assert obj is not None
    assert obj["choices"][0]["delta"]["content"] == "round-trip"


def test_parse_done_returns_none() -> None:
    assert parse_sse_event("data: [DONE]\n\n") is None


def test_max_chunk_bytes_enforced() -> None:
    # 256-byte cap — normal tiny deltas pass, a huge content chunk trips it.
    stream = SSEChatStream("aurelius-test", max_chunk_bytes=256)
    assert stream.emit_role_delta()  # tiny frame fits
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        stream.emit_content_delta("x" * 4096)


def test_unicode_content_preserved() -> None:
    stream = SSEChatStream("aurelius-test")
    text = "héllo 🌍 \u4e2d\u6587"
    frame = stream.emit_content_delta(text)
    obj = parse_sse_event(frame)
    assert obj is not None
    assert obj["choices"][0]["delta"]["content"] == text


def test_empty_content_delta_still_valid_json() -> None:
    stream = SSEChatStream("aurelius-test")
    frame = stream.emit_content_delta("")
    obj = parse_sse_event(frame)
    assert obj is not None
    assert obj["choices"][0]["delta"]["content"] == ""


def test_concatenated_frames_parseable() -> None:
    stream = SSEChatStream("aurelius-test")
    buf = (
        stream.emit_role_delta()
        + stream.emit_content_delta("a")
        + stream.emit_content_delta("b")
        + stream.emit_finish("stop")
        + stream.emit_done()
    )
    records = [r for r in buf.split("\n\n") if r]
    parsed = [parse_sse_event(r + "\n\n") for r in records]
    assert parsed[-1] is None
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "a"
    assert parsed[2]["choices"][0]["delta"]["content"] == "b"
    assert parsed[3]["choices"][0]["finish_reason"] == "stop"


def test_determinism_same_inputs_same_frame() -> None:
    s1 = SSEChatStream("aurelius-test")
    s2 = SSEChatStream("aurelius-test")
    # Stream id / created differ between instances, but for a single stream
    # the same input must yield the same frame bytes.
    f_a = s1.emit_content_delta("tok")
    f_b = s1.emit_content_delta("tok")
    assert f_a == f_b
    # Different instances keep their own identity (not asserted equal) but
    # payload *shape* is identical.
    obj_a = parse_sse_event(f_a)
    obj_c = parse_sse_event(s2.emit_content_delta("tok"))
    assert set(obj_a) == set(obj_c)


def test_json_keys_match_openai_shape() -> None:
    stream = SSEChatStream("aurelius-test")
    obj = parse_sse_event(stream.emit_role_delta())
    assert set(obj.keys()) == {"id", "object", "created", "model", "choices"}
    assert obj["object"] == "chat.completion.chunk"
    choice_keys = set(obj["choices"][0].keys())
    assert choice_keys == {"index", "delta", "finish_reason"}


def test_created_timestamp_is_int() -> None:
    stream = SSEChatStream("aurelius-test")
    obj = parse_sse_event(stream.emit_role_delta())
    assert isinstance(obj["created"], int)


def test_sse_parse_error_on_malformed() -> None:
    with pytest.raises(SSEParseError):
        parse_sse_event("event: foo\ndata: {}\n\n")
    with pytest.raises(SSEParseError):
        parse_sse_event("data: {not json}\n\n")
    with pytest.raises(SSEParseError):
        parse_sse_event("data: \n\n")
    with pytest.raises(SSEParseError):
        parse_sse_event(123)  # type: ignore[arg-type]


def test_dataclass_defaults() -> None:
    chunk = ChatCompletionChunk(id="x")
    assert chunk.object_type == "chat.completion.chunk"
    assert chunk.choices == []
    cd = ChoiceDelta(index=0, delta={"content": "a"})
    assert cd.finish_reason is None


def test_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        SSEChatStream("")
    with pytest.raises(ValueError):
        SSEChatStream("m", max_chunk_bytes=0)
