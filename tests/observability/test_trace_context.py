"""Tests for TraceContext."""

from __future__ import annotations

import threading

import pytest

from src.observability.trace_context import TraceContext, _gen_id


class TestTraceContext:
    def test_new_without_parent(self) -> None:
        ctx = TraceContext.new()
        assert ctx.trace_id is not None
        assert ctx.span_id is not None
        assert ctx.parent_span_id is None

    def test_new_with_parent(self) -> None:
        parent = TraceContext.new()
        child = TraceContext.new(parent=parent)
        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id

    def test_child_creates_new_span_with_parent(self) -> None:
        parent = TraceContext.new()
        child = parent.child()
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id

    def test_current_returns_none_initially(self) -> None:
        TraceContext.clear_current()
        assert TraceContext.current() is None

    def test_set_and_get_current(self) -> None:
        TraceContext.clear_current()
        ctx = TraceContext.new()
        TraceContext.set_current(ctx)
        assert TraceContext.current() is ctx
        TraceContext.clear_current()

    def test_clear_current(self) -> None:
        TraceContext.set_current(TraceContext.new())
        TraceContext.clear_current()
        assert TraceContext.current() is None

    def test_scope_sets_current(self) -> None:
        TraceContext.clear_current()
        ctx = TraceContext.new()
        with TraceContext.scope(ctx) as yielded:
            assert yielded is ctx
            assert TraceContext.current() is ctx
        assert TraceContext.current() is None

    def test_scope_creates_new_context_if_none_given(self) -> None:
        TraceContext.clear_current()
        with TraceContext.scope() as ctx:
            assert ctx is not None
            assert TraceContext.current() is ctx
        assert TraceContext.current() is None

    def test_scope_restores_previous_on_exception(self) -> None:
        TraceContext.clear_current()
        previous = TraceContext.new()
        TraceContext.set_current(previous)
        with pytest.raises(ValueError):
            with TraceContext.scope():
                raise ValueError("boom")
        assert TraceContext.current() is previous
        TraceContext.clear_current()

    def test_scope_restores_previous_on_normal_exit(self) -> None:
        TraceContext.clear_current()
        previous = TraceContext.new()
        TraceContext.set_current(previous)
        with TraceContext.scope():
            pass
        assert TraceContext.current() is previous
        TraceContext.clear_current()

    def test_to_dict(self) -> None:
        ctx = TraceContext.new()
        d = ctx.to_dict()
        assert d["trace_id"] == ctx.trace_id
        assert d["span_id"] == ctx.span_id
        assert d["parent_span_id"] == ctx.parent_span_id

    def test_to_dict_with_parent(self) -> None:
        parent = TraceContext.new()
        child = TraceContext.new(parent=parent)
        d = child.to_dict()
        assert d["trace_id"] == parent.trace_id
        assert d["span_id"] == child.span_id
        assert d["parent_span_id"] == parent.span_id

    def test_immutable_dataclass(self) -> None:
        ctx = TraceContext.new()
        with pytest.raises(AttributeError):
            ctx.trace_id = "x"  # type: ignore[misc]

    def test_slots_present(self) -> None:
        ctx = TraceContext.new()
        with pytest.raises(AttributeError):
            ctx.__dict__  # type: ignore[unused-ignore]

    def test_ids_are_16_chars(self) -> None:
        for _ in range(100):
            tid = _gen_id()
            assert len(tid) == 16

    def test_ids_are_unique(self) -> None:
        ids = set()
        for _ in range(1000):
            ids.add(_gen_id())
        assert len(ids) == 1000


class TestTraceContextThreading:
    def test_thread_local_isolation(self) -> None:
        results: list[str | None] = []

        def worker() -> None:
            ctx = TraceContext.new()
            TraceContext.set_current(ctx)
            results.append(TraceContext.current().trace_id if TraceContext.current() else None)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None
        assert results[0] != results[1]


class TestGenId:
    def test_gen_id_returns_hex_string(self) -> None:
        tid = _gen_id()
        assert isinstance(tid, str)
        int(tid, 16)  # must be valid hex

    def test_gen_id_length(self) -> None:
        tid = _gen_id()
        assert len(tid) == 16
