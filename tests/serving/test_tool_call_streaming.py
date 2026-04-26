"""Tests for src/serving/tool_call_streaming.py (~45 tests)."""

from src.serving.tool_call_streaming import (
    TOOL_CALL_ACCUMULATOR_REGISTRY,
    ToolCallBuffer,
    ToolCallState,
    ToolCallStreamAccumulator,
)

# ---------------------------------------------------------------------------
# ToolCallState enum
# ---------------------------------------------------------------------------


class TestToolCallStateEnum:
    def test_pending_value(self):
        assert ToolCallState.PENDING == "pending"

    def test_streaming_value(self):
        assert ToolCallState.STREAMING == "streaming"

    def test_complete_value(self):
        assert ToolCallState.COMPLETE == "complete"

    def test_error_value(self):
        assert ToolCallState.ERROR == "error"

    def test_is_str_subclass(self):
        assert isinstance(ToolCallState.PENDING, str)

    def test_all_four_members(self):
        members = {m.value for m in ToolCallState}
        assert members == {"pending", "streaming", "complete", "error"}


# ---------------------------------------------------------------------------
# ToolCallBuffer – initial state
# ---------------------------------------------------------------------------


class TestToolCallBufferInit:
    def test_initial_state_is_pending(self):
        buf = ToolCallBuffer("tc-1", "my_fn")
        assert buf.state == ToolCallState.PENDING

    def test_tool_call_id_property(self):
        buf = ToolCallBuffer("abc", "fn")
        assert buf.tool_call_id == "abc"

    def test_function_name_property(self):
        buf = ToolCallBuffer("id", "do_thing")
        assert buf.function_name == "do_thing"

    def test_raw_arguments_empty_initially(self):
        buf = ToolCallBuffer("x", "y")
        assert buf.raw_arguments == ""


# ---------------------------------------------------------------------------
# ToolCallBuffer – append_argument_delta
# ---------------------------------------------------------------------------


class TestToolCallBufferAppend:
    def test_single_delta_accumulates(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"key":')
        assert buf.raw_arguments == '{"key":'

    def test_multiple_deltas_accumulate(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"key":')
        buf.append_argument_delta('"value"}')
        assert buf.raw_arguments == '{"key":"value"}'

    def test_state_becomes_streaming_after_first_delta(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("x")
        assert buf.state == ToolCallState.STREAMING

    def test_state_remains_streaming_after_second_delta(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("a")
        buf.append_argument_delta("b")
        assert buf.state == ToolCallState.STREAMING

    def test_empty_delta_still_transitions_to_streaming(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("")
        # An empty string is still a delta event; state should move.
        assert buf.state == ToolCallState.STREAMING


# ---------------------------------------------------------------------------
# ToolCallBuffer – finalize
# ---------------------------------------------------------------------------


class TestToolCallBufferFinalize:
    def test_finalize_valid_json_returns_dict(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"a": 1}')
        result = buf.finalize()
        assert result == {"a": 1}

    def test_finalize_valid_json_sets_complete(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"x": true}')
        buf.finalize()
        assert buf.state == ToolCallState.COMPLETE

    def test_finalize_invalid_json_returns_none(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("{broken")
        result = buf.finalize()
        assert result is None

    def test_finalize_invalid_json_sets_error(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("{broken")
        buf.finalize()
        assert buf.state == ToolCallState.ERROR

    def test_finalize_empty_string_returns_none(self):
        buf = ToolCallBuffer("t1", "fn")
        result = buf.finalize()
        assert result is None

    def test_finalize_nested_dict(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"outer": {"inner": 42}}')
        result = buf.finalize()
        assert result == {"outer": {"inner": 42}}

    def test_finalize_list_returns_none(self):
        # JSON array is not a dict – json.loads would return a list;
        # finalize returns None only on JSONDecodeError/ValueError.
        # A list is valid JSON but finalize must still return the parsed value.
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta("[1, 2, 3]")
        result = buf.finalize()
        # json.loads("[1,2,3]") succeeds → returns list, state=COMPLETE.
        assert result == [1, 2, 3]
        assert buf.state == ToolCallState.COMPLETE

    def test_raw_arguments_unchanged_after_finalize(self):
        buf = ToolCallBuffer("t1", "fn")
        buf.append_argument_delta('{"z": 0}')
        buf.finalize()
        assert buf.raw_arguments == '{"z": 0}'


# ---------------------------------------------------------------------------
# ToolCallStreamAccumulator
# ---------------------------------------------------------------------------


class TestToolCallStreamAccumulator:
    def test_start_tool_call_returns_buffer(self):
        acc = ToolCallStreamAccumulator()
        buf = acc.start_tool_call("tc-1", "fn")
        assert isinstance(buf, ToolCallBuffer)

    def test_start_tool_call_registers_buffer(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        assert acc.get_buffer("tc-1") is not None

    def test_get_buffer_unknown_id_returns_none(self):
        acc = ToolCallStreamAccumulator()
        assert acc.get_buffer("nonexistent") is None

    def test_append_delta_unknown_id_returns_false(self):
        acc = ToolCallStreamAccumulator()
        result = acc.append_delta("ghost", "data")
        assert result is False

    def test_append_delta_known_id_returns_true(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        result = acc.append_delta("tc-1", '{"k":')
        assert result is True

    def test_append_delta_accumulates_in_buffer(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        acc.append_delta("tc-1", '{"k":')
        acc.append_delta("tc-1", '"v"}')
        assert acc.get_buffer("tc-1").raw_arguments == '{"k":"v"}'

    def test_finalize_all_returns_only_successful_dicts(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-ok", "fn")
        acc.append_delta("tc-ok", '{"a": 1}')
        acc.start_tool_call("tc-bad", "fn")
        acc.append_delta("tc-bad", "{broken")
        results = acc.finalize_all()
        assert results == [{"a": 1}]

    def test_finalize_all_empty_accumulator(self):
        acc = ToolCallStreamAccumulator()
        assert acc.finalize_all() == []

    def test_pending_count_before_finalize(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        acc.start_tool_call("tc-2", "fn")
        assert acc.pending_count() == 2

    def test_pending_count_after_finalize_all(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        acc.append_delta("tc-1", '{"x": 1}')
        assert acc.pending_count() == 1
        acc.finalize_all()
        assert acc.pending_count() == 0

    def test_pending_count_includes_streaming_not_complete(self):
        acc = ToolCallStreamAccumulator()
        acc.start_tool_call("tc-1", "fn")
        acc.append_delta("tc-1", '{"partial":')  # streaming, not finalised
        assert acc.pending_count() == 1

    def test_multiple_successful_finalizations(self):
        acc = ToolCallStreamAccumulator()
        for i in range(3):
            acc.start_tool_call(f"tc-{i}", "fn")
            acc.append_delta(f"tc-{i}", f'{{"i": {i}}}')
        results = acc.finalize_all()
        assert len(results) == 3

    def test_registry_contains_default(self):
        assert "default" in TOOL_CALL_ACCUMULATOR_REGISTRY

    def test_registry_default_is_accumulator_class(self):
        cls = TOOL_CALL_ACCUMULATOR_REGISTRY["default"]
        assert cls is ToolCallStreamAccumulator

    def test_registry_default_is_instantiable(self):
        cls = TOOL_CALL_ACCUMULATOR_REGISTRY["default"]
        acc = cls()
        assert isinstance(acc, ToolCallStreamAccumulator)
