"""Tests for src/protocol/streaming_protocol.py (~45 tests)."""

from src.protocol.streaming_protocol import (
    STREAMING_PROTOCOL,
    StreamEvent,
    StreamEventType,
    StreamingProtocol,
)


class TestStreamEventTypeEnum:
    def test_enum_count(self):
        assert len(StreamEventType) == 7

    def test_text_delta_value(self):
        assert StreamEventType.TEXT_DELTA == "text_delta"

    def test_tool_call_start_value(self):
        assert StreamEventType.TOOL_CALL_START == "tool_call_start"

    def test_tool_call_delta_value(self):
        assert StreamEventType.TOOL_CALL_DELTA == "tool_call_delta"

    def test_tool_call_end_value(self):
        assert StreamEventType.TOOL_CALL_END == "tool_call_end"

    def test_done_value(self):
        assert StreamEventType.DONE == "done"

    def test_error_value(self):
        assert StreamEventType.ERROR == "error"

    def test_ping_value(self):
        assert StreamEventType.PING == "ping"

    def test_is_str_subclass(self):
        assert isinstance(StreamEventType.TEXT_DELTA, str)


class TestStreamEvent:
    def test_id_auto_generated(self):
        ev = StreamEvent(event_type=StreamEventType.PING)
        assert ev.id is not None

    def test_id_is_8_chars(self):
        ev = StreamEvent(event_type=StreamEventType.PING)
        assert len(ev.id) == 8

    def test_id_is_hex(self):
        ev = StreamEvent(event_type=StreamEventType.PING)
        int(ev.id, 16)

    def test_two_events_have_different_ids(self):
        e1 = StreamEvent(event_type=StreamEventType.PING)
        e2 = StreamEvent(event_type=StreamEventType.PING)
        assert e1.id != e2.id

    def test_data_default_empty_string(self):
        ev = StreamEvent(event_type=StreamEventType.PING)
        assert ev.data == ""

    def test_sequence_default_zero(self):
        ev = StreamEvent(event_type=StreamEventType.PING)
        assert ev.sequence == 0

    def test_event_type_stored(self):
        ev = StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hello")
        assert ev.event_type == StreamEventType.TEXT_DELTA

    def test_data_stored(self):
        ev = StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hello")
        assert ev.data == "hello"

    def test_sequence_can_be_set(self):
        ev = StreamEvent(event_type=StreamEventType.TEXT_DELTA, sequence=5)
        assert ev.sequence == 5


class TestStreamingProtocol:
    def setup_method(self):
        self.proto = StreamingProtocol()

    def test_emit_returns_stream_event(self):
        ev = self.proto.emit(StreamEventType.PING)
        assert isinstance(ev, StreamEvent)

    def test_emit_event_type_matches(self):
        ev = self.proto.emit(StreamEventType.TEXT_DELTA, "hi")
        assert ev.event_type == StreamEventType.TEXT_DELTA

    def test_emit_data_stored(self):
        ev = self.proto.emit(StreamEventType.TEXT_DELTA, "hello world")
        assert ev.data == "hello world"

    def test_emit_first_sequence_is_zero(self):
        ev = self.proto.emit(StreamEventType.PING)
        assert ev.sequence == 0

    def test_emit_auto_increments_sequence(self):
        e0 = self.proto.emit(StreamEventType.PING)
        e1 = self.proto.emit(StreamEventType.PING)
        e2 = self.proto.emit(StreamEventType.PING)
        assert e0.sequence == 0
        assert e1.sequence == 1
        assert e2.sequence == 2

    def test_emit_appends_to_log(self):
        self.proto.emit(StreamEventType.PING)
        self.proto.emit(StreamEventType.TEXT_DELTA, "x")
        assert len(self.proto.event_log()) == 2

    def test_to_sse_contains_id(self):
        ev = self.proto.emit(StreamEventType.PING)
        sse = self.proto.to_sse(ev)
        assert f"id: {ev.id}" in sse

    def test_to_sse_contains_event(self):
        ev = self.proto.emit(StreamEventType.TEXT_DELTA, "x")
        sse = self.proto.to_sse(ev)
        assert "event:" in sse

    def test_to_sse_contains_event_type_value(self):
        ev = self.proto.emit(StreamEventType.TEXT_DELTA, "x")
        sse = self.proto.to_sse(ev)
        assert "text_delta" in sse

    def test_to_sse_contains_data(self):
        ev = self.proto.emit(StreamEventType.TEXT_DELTA, "my data")
        sse = self.proto.to_sse(ev)
        assert "data:" in sse
        assert "my data" in sse

    def test_to_sse_ends_with_double_newline(self):
        ev = self.proto.emit(StreamEventType.PING)
        sse = self.proto.to_sse(ev)
        assert sse.endswith("\n\n")

    def test_to_sse_format_structure(self):
        ev = self.proto.emit(StreamEventType.DONE)
        sse = self.proto.to_sse(ev)
        lines = sse.split("\n")
        assert lines[0].startswith("id:")
        assert lines[1].startswith("event:")
        assert lines[2].startswith("data:")

    def test_accumulate_text_joins_text_delta_data(self):
        events = [
            StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="Hello"),
            StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=" World"),
        ]
        result = self.proto.accumulate_text(events)
        assert result == "Hello World"

    def test_accumulate_text_ignores_non_text_delta(self):
        events = [
            StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hello"),
            StreamEvent(event_type=StreamEventType.PING, data="ignored"),
            StreamEvent(event_type=StreamEventType.DONE, data="also ignored"),
        ]
        result = self.proto.accumulate_text(events)
        assert result == "hello"

    def test_accumulate_text_empty_list(self):
        assert self.proto.accumulate_text([]) == ""

    def test_accumulate_text_no_text_delta_returns_empty(self):
        events = [
            StreamEvent(event_type=StreamEventType.DONE),
            StreamEvent(event_type=StreamEventType.PING),
        ]
        assert self.proto.accumulate_text(events) == ""

    def test_is_done_true_when_done_event_present(self):
        events = [
            StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hi"),
            StreamEvent(event_type=StreamEventType.DONE),
        ]
        assert self.proto.is_done(events) is True

    def test_is_done_false_when_no_done_event(self):
        events = [
            StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hi"),
            StreamEvent(event_type=StreamEventType.PING),
        ]
        assert self.proto.is_done(events) is False

    def test_is_done_false_for_empty_list(self):
        assert self.proto.is_done([]) is False

    def test_event_log_returns_all_emitted(self):
        self.proto.emit(StreamEventType.TEXT_DELTA, "a")
        self.proto.emit(StreamEventType.DONE)
        log = self.proto.event_log()
        assert len(log) == 2

    def test_event_log_returns_list(self):
        assert isinstance(self.proto.event_log(), list)

    def test_event_log_is_copy_not_reference(self):
        self.proto.emit(StreamEventType.PING)
        log1 = self.proto.event_log()
        self.proto.emit(StreamEventType.PING)
        log2 = self.proto.event_log()
        assert len(log1) == 1
        assert len(log2) == 2

    def test_reset_clears_log(self):
        self.proto.emit(StreamEventType.PING)
        self.proto.reset()
        assert self.proto.event_log() == []

    def test_reset_resets_sequence_counter(self):
        self.proto.emit(StreamEventType.PING)
        self.proto.emit(StreamEventType.PING)
        self.proto.reset()
        ev = self.proto.emit(StreamEventType.PING)
        assert ev.sequence == 0

    def test_after_reset_sequence_starts_from_zero(self):
        for _ in range(5):
            self.proto.emit(StreamEventType.PING)
        self.proto.reset()
        e0 = self.proto.emit(StreamEventType.TEXT_DELTA, "x")
        e1 = self.proto.emit(StreamEventType.TEXT_DELTA, "y")
        assert e0.sequence == 0
        assert e1.sequence == 1

    def test_streaming_protocol_singleton_exists(self):
        assert STREAMING_PROTOCOL is not None
        assert isinstance(STREAMING_PROTOCOL, StreamingProtocol)
