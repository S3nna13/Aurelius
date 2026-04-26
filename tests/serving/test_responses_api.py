"""Unit tests for src/serving/responses_api.py — Responses API shape."""

from __future__ import annotations

from src.serving.responses_api import (
    RESPONSE_COMPLETED,
    RESPONSE_CREATED,
    RESPONSES_API_REGISTRY,
    InputItem,
    ResponseOutputItem,
    ResponsesAPIHandler,
    ResponsesAPIModel,
    ResponsesAPIRequest,
    ResponsesAPIResponse,
    ResponsesAPIValidator,
    ResponseTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_request(**overrides) -> ResponsesAPIRequest:
    defaults = dict(
        model="aurelius-base",
        input=[InputItem(role="user", content="Hello")],
    )
    defaults.update(overrides)
    return ResponsesAPIRequest(**defaults)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclassConstruction:
    def test_request_defaults(self):
        req = _valid_request()
        assert req.temperature == 1.0
        assert req.stream is False
        assert req.store is True
        assert req.tools is None
        assert req.max_output_tokens is None
        assert req.previous_response_id is None
        assert req.reasoning is None

    def test_input_item_defaults(self):
        item = InputItem(role="user", content="hi")
        assert item.input_type == "message"

    def test_input_item_tool_result(self):
        item = InputItem(role="tool", content="result", input_type="tool_result")
        assert item.role == "tool"
        assert item.input_type == "tool_result"

    def test_response_tool_construction(self):
        tool = ResponseTool(
            type="function",
            name="my_fn",
            description="Does something",
            input_schema={"type": "object"},
        )
        assert tool.type == "function"
        assert tool.name == "my_fn"

    def test_response_output_item_default_status(self):
        item = ResponseOutputItem(id="x", type="message")
        assert item.status == "in_progress"

    def test_response_output_item_tool_call(self):
        item = ResponseOutputItem(
            id="y",
            type="tool_call",
            tool_name="search",
            tool_input={"query": "test"},
        )
        assert item.tool_name == "search"
        assert item.tool_input == {"query": "test"}


# ---------------------------------------------------------------------------
# ResponsesAPIModel enum
# ---------------------------------------------------------------------------


class TestResponsesAPIModel:
    def test_enum_has_aurelius_base(self):
        assert ResponsesAPIModel.base.value == "aurelius-base"

    def test_enum_has_all_four_models(self):
        values = {m.value for m in ResponsesAPIModel}
        assert values == {
            "aurelius-base",
            "aurelius-chat",
            "aurelius-coding",
            "aurelius-long",
        }


# ---------------------------------------------------------------------------
# Validator — request
# ---------------------------------------------------------------------------


class TestValidatorRequest:
    def setup_method(self):
        self.v = ResponsesAPIValidator()

    def test_valid_request_returns_empty(self):
        errors = self.v.validate_request(_valid_request())
        assert errors == []

    def test_empty_model_returns_error(self):
        req = _valid_request(model="")
        errors = self.v.validate_request(req)
        assert any("model must not be empty" in e for e in errors)

    def test_empty_input_returns_error(self):
        req = _valid_request(input=[])
        errors = self.v.validate_request(req)
        assert errors  # at least one error about empty input

    def test_temperature_too_high(self):
        req = _valid_request(temperature=3.0)
        errors = self.v.validate_request(req)
        assert errors  # temperature out of range

    def test_temperature_negative(self):
        req = _valid_request(temperature=-0.1)
        errors = self.v.validate_request(req)
        assert errors

    def test_temperature_boundary_zero(self):
        assert self.v.validate_request(_valid_request(temperature=0.0)) == []

    def test_temperature_boundary_two(self):
        assert self.v.validate_request(_valid_request(temperature=2.0)) == []

    def test_max_output_tokens_zero_returns_error(self):
        req = _valid_request(max_output_tokens=0)
        errors = self.v.validate_request(req)
        assert errors

    def test_max_output_tokens_positive_ok(self):
        req = _valid_request(max_output_tokens=512)
        assert self.v.validate_request(req) == []

    def test_tool_with_empty_name_returns_error(self):
        tool = ResponseTool(type="function", name="", description="oops")
        req = _valid_request(tools=[tool])
        errors = self.v.validate_request(req)
        assert errors


# ---------------------------------------------------------------------------
# Validator — response
# ---------------------------------------------------------------------------


class TestValidatorResponse:
    def setup_method(self):
        self.v = ResponsesAPIValidator()
        self.handler = ResponsesAPIHandler()

    def test_valid_response_returns_empty(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert self.v.validate_response(resp) == []

    def test_empty_id_returns_error(self):
        import time

        resp = ResponsesAPIResponse(
            id="",
            model="aurelius-base",
            status="completed",
            output=[],
            usage={},
            created_at=time.time(),
        )
        errors = self.v.validate_response(resp)
        assert errors


# ---------------------------------------------------------------------------
# Handler — create_response
# ---------------------------------------------------------------------------


class TestHandlerCreateResponse:
    def setup_method(self):
        self.handler = ResponsesAPIHandler()

    def test_create_response_returns_completed_status(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert resp.status == "completed"

    def test_response_id_non_empty(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert isinstance(resp.id, str) and resp.id

    def test_response_object_field(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert resp.object == "response"

    def test_response_has_one_output_item(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert len(resp.output) == 1

    def test_response_output_content_references_model(self):
        req = _valid_request(model="aurelius-coding")
        resp = self.handler.create_response(req)
        assert "aurelius-coding" in resp.output[0].content

    def test_usage_has_required_keys(self):
        req = _valid_request()
        resp = self.handler.create_response(req)
        assert "input_tokens" in resp.usage
        assert "output_tokens" in resp.usage
        assert resp.usage["output_tokens"] == 10

    def test_counter_increments(self):
        before = ResponsesAPIHandler.RESPONSE_COUNTER
        self.handler.create_response(_valid_request())
        self.handler.create_response(_valid_request())
        assert ResponsesAPIHandler.RESPONSE_COUNTER >= before + 2


# ---------------------------------------------------------------------------
# Handler — stream_events
# ---------------------------------------------------------------------------


class TestHandlerStreamEvents:
    def setup_method(self):
        self.handler = ResponsesAPIHandler()

    def test_stream_yields_at_least_three_events(self):
        events = list(self.handler.stream_events(_valid_request()))
        assert len(events) >= 3

    def test_first_event_is_response_created(self):
        events = list(self.handler.stream_events(_valid_request()))
        assert events[0].event_type == RESPONSE_CREATED

    def test_last_event_is_response_completed(self):
        events = list(self.handler.stream_events(_valid_request()))
        assert events[-1].event_type == RESPONSE_COMPLETED

    def test_sequence_numbers_are_ordered(self):
        events = list(self.handler.stream_events(_valid_request()))
        seq_nums = [e.sequence_number for e in events]
        assert seq_nums == sorted(seq_nums)

    def test_events_have_data_dicts(self):
        for event in self.handler.stream_events(_valid_request()):
            assert isinstance(event.data, dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_responses_api_registry_contains_default(self):
        assert "default" in RESPONSES_API_REGISTRY

    def test_responses_api_registry_default_is_handler(self):
        assert RESPONSES_API_REGISTRY["default"] is ResponsesAPIHandler
