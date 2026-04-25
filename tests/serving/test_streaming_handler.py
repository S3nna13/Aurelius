"""Tests for streaming_handler.py."""

from __future__ import annotations

import time

import pytest

from src.serving.streaming_handler import StreamChunk, StreamSession, StreamingHandler


def make_handler(**kwargs) -> StreamingHandler:
    return StreamingHandler(**kwargs)


def test_create_session_returns_session():
    h = make_handler()
    s = h.create_session("req-1")
    assert isinstance(s, StreamSession)
    assert s.request_id == "req-1"
    assert s.session_id


def test_create_session_increments_active_count():
    h = make_handler()
    h.create_session("r1")
    h.create_session("r2")
    assert h.active_sessions() == 2


def test_session_limit_raises():
    h = make_handler(max_sessions=2)
    h.create_session("r1")
    h.create_session("r2")
    with pytest.raises(RuntimeError):
        h.create_session("r3")


def test_push_chunk_returns_chunk():
    h = make_handler()
    s = h.create_session("r1")
    chunk = h.push_chunk(s.session_id, "hello world")
    assert isinstance(chunk, StreamChunk)
    assert chunk.text == "hello world"
    assert chunk.chunk_id == 0


def test_push_chunk_increments_ids():
    h = make_handler()
    s = h.create_session("r1")
    c1 = h.push_chunk(s.session_id, "a")
    c2 = h.push_chunk(s.session_id, "b")
    assert c1.chunk_id == 0
    assert c2.chunk_id == 1


def test_push_chunk_token_count_nonzero():
    h = make_handler()
    s = h.create_session("r1")
    c = h.push_chunk(s.session_id, "one two three")
    assert c.token_count >= 1


def test_push_chunk_final_marks_session_complete():
    h = make_handler()
    s = h.create_session("r1")
    h.push_chunk(s.session_id, "done", is_final=True, finish_reason="stop")
    assert h.get_session(s.session_id).is_complete is True


def test_push_chunk_finish_reason_propagated():
    h = make_handler()
    s = h.create_session("r1")
    c = h.push_chunk(s.session_id, "x", is_final=True, finish_reason="length")
    assert c.finish_reason == "length"


def test_get_session_returns_none_for_unknown():
    h = make_handler()
    assert h.get_session("nonexistent") is None


def test_get_session_returns_correct_session():
    h = make_handler()
    s = h.create_session("r1")
    assert h.get_session(s.session_id) is s


def test_collect_concatenates_text():
    h = make_handler()
    s = h.create_session("r1")
    h.push_chunk(s.session_id, "foo")
    h.push_chunk(s.session_id, "bar")
    assert h.collect(s.session_id) == "foobar"


def test_collect_empty_session():
    h = make_handler()
    s = h.create_session("r1")
    assert h.collect(s.session_id) == ""


def test_finalize_marks_complete():
    h = make_handler()
    s = h.create_session("r1")
    result = h.finalize(s.session_id)
    assert result.is_complete is True


def test_finalize_already_complete_raises():
    h = make_handler()
    s = h.create_session("r1")
    h.finalize(s.session_id)
    with pytest.raises(ValueError):
        h.finalize(s.session_id)


def test_prune_removes_old_completed():
    h = make_handler()
    s = h.create_session("r1")
    h.finalize(s.session_id)
    s._created_at_override = None
    session = h.get_session(s.session_id)
    session.created_at = time.time() - 400
    removed = h.prune_completed(max_age_s=300.0)
    assert removed == 1
    assert h.active_sessions() == 0


def test_prune_keeps_recent_completed():
    h = make_handler()
    s = h.create_session("r1")
    h.finalize(s.session_id)
    removed = h.prune_completed(max_age_s=300.0)
    assert removed == 0
    assert h.active_sessions() == 1


def test_prune_keeps_incomplete_old():
    h = make_handler()
    s = h.create_session("r1")
    session = h.get_session(s.session_id)
    session.created_at = time.time() - 400
    removed = h.prune_completed(max_age_s=300.0)
    assert removed == 0
    assert h.active_sessions() == 1


def test_iter_chunks_returns_list():
    h = make_handler()
    s = h.create_session("r1")
    h.push_chunk(s.session_id, "a")
    h.push_chunk(s.session_id, "b")
    chunks = h.iter_chunks(s.session_id)
    assert isinstance(chunks, list)
    assert len(chunks) == 2


def test_total_tokens_accumulates():
    h = make_handler()
    s = h.create_session("r1")
    h.push_chunk(s.session_id, "one two")
    h.push_chunk(s.session_id, "three four five")
    session = h.get_session(s.session_id)
    assert session.total_tokens > 0


def test_active_sessions_decreases_after_prune():
    h = make_handler()
    s1 = h.create_session("r1")
    s2 = h.create_session("r2")
    h.finalize(s1.session_id)
    h.get_session(s1.session_id).created_at = time.time() - 400
    h.prune_completed(max_age_s=300.0)
    assert h.active_sessions() == 1
