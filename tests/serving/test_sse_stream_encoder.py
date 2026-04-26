"""Tests for sse_stream_encoder."""

from __future__ import annotations

import pytest

from src.serving.sse_stream_encoder import SSEStreamEncoder, split_sse_records


def test_encode_event_basic():
    enc = SSEStreamEncoder()
    raw = enc.encode_event(data="hello")
    assert raw.endswith(b"\n\n")
    assert b"data: hello" in raw


def test_encode_event_multiline_data():
    enc = SSEStreamEncoder()
    raw = enc.encode_event(data="a\nb")
    assert b"data: a" in raw and b"data: b" in raw


def test_encode_event_with_event_and_id():
    enc = SSEStreamEncoder()
    raw = enc.encode_event(data="x", event="delta", event_id="7")
    assert b"event: delta" in raw
    assert b"id: 7" in raw


def test_rejects_newline_in_event():
    with pytest.raises(ValueError):
        SSEStreamEncoder().encode_event(data="ok", event="bad\nname")


def test_rejects_newline_in_id():
    with pytest.raises(ValueError):
        SSEStreamEncoder().encode_event(data="ok", event_id="1\n2")


def test_rejects_nul_in_data():
    with pytest.raises(ValueError):
        SSEStreamEncoder().encode_event(data="a\x00b")


def test_encode_comment():
    enc = SSEStreamEncoder()
    raw = enc.encode_comment("ping")
    assert raw == b": ping\n"


def test_split_sse_records():
    enc = SSEStreamEncoder()
    blob = enc.encode_event(data="one") + enc.encode_event(data="two")
    recs = split_sse_records(blob)
    assert len(recs) == 2


def test_type_errors():
    with pytest.raises(TypeError):
        SSEStreamEncoder().encode_event(data=123)  # type: ignore[arg-type]


def test_utf8_errors_param():
    enc = SSEStreamEncoder(utf8_errors="replace")
    raw = enc.encode_event(data="café")
    assert isinstance(raw, bytes)
    assert "café".encode() in raw
