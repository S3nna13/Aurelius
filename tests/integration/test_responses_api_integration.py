"""Integration tests for the Responses API shape within the Aurelius serving layer.

Verifies that:
- The Responses API is correctly registered in API_SHAPE_REGISTRY
- A full request → create_response → validate_response round-trip passes all checks
- A streaming round-trip yields properly ordered events
"""

from __future__ import annotations

import src.serving  # noqa: F401  — triggers registry population


from src.serving import API_SHAPE_REGISTRY
from src.serving.responses_api import (
    RESPONSE_COMPLETED,
    RESPONSE_CREATED,
    RESPONSES_API_REGISTRY,
    InputItem,
    ResponsesAPIHandler,
    ResponsesAPIValidator,
    ResponsesAPIRequest,
)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_responses_key_in_api_shape_registry(self):
        assert "responses" in API_SHAPE_REGISTRY

    def test_api_shape_registry_responses_is_handler(self):
        # Compare by module/name rather than object identity so this stays
        # stable even if another test reloads the serving package.
        handler_cls = API_SHAPE_REGISTRY["responses"]
        assert handler_cls.__module__ == ResponsesAPIHandler.__module__
        assert handler_cls.__name__ == ResponsesAPIHandler.__name__

    def test_responses_api_registry_accessible(self):
        assert RESPONSES_API_REGISTRY is not None
        assert "default" in RESPONSES_API_REGISTRY


# ---------------------------------------------------------------------------
# Round-trip integration
# ---------------------------------------------------------------------------

class TestRoundTripIntegration:
    def _make_request(self, **kwargs) -> ResponsesAPIRequest:
        defaults = dict(
            model="aurelius-chat",
            input=[InputItem(role="user", content="Integration test message")],
        )
        defaults.update(kwargs)
        return ResponsesAPIRequest(**defaults)

    def test_full_roundtrip_passes_validation(self):
        handler = ResponsesAPIHandler()
        validator = ResponsesAPIValidator()

        req = self._make_request()
        req_errors = validator.validate_request(req)
        assert req_errors == [], f"Request validation errors: {req_errors}"

        resp = handler.create_response(req)
        resp_errors = validator.validate_response(resp)
        assert resp_errors == [], f"Response validation errors: {resp_errors}"

    def test_roundtrip_response_status_completed(self):
        handler = ResponsesAPIHandler()
        req = self._make_request()
        resp = handler.create_response(req)
        assert resp.status == "completed"

    def test_roundtrip_response_model_matches(self):
        handler = ResponsesAPIHandler()
        req = self._make_request(model="aurelius-coding")
        resp = handler.create_response(req)
        assert resp.model == "aurelius-coding"

    def test_roundtrip_output_non_empty(self):
        handler = ResponsesAPIHandler()
        req = self._make_request()
        resp = handler.create_response(req)
        assert len(resp.output) > 0

    def test_roundtrip_response_id_unique(self):
        handler = ResponsesAPIHandler()
        req = self._make_request()
        ids = {handler.create_response(req).id for _ in range(5)}
        assert len(ids) == 5  # all unique


# ---------------------------------------------------------------------------
# Streaming integration
# ---------------------------------------------------------------------------

class TestStreamingIntegration:
    def _make_request(self) -> ResponsesAPIRequest:
        return ResponsesAPIRequest(
            model="aurelius-base",
            input=[InputItem(role="user", content="stream me")],
            stream=True,
        )

    def test_stream_first_event_is_response_created(self):
        handler = ResponsesAPIHandler()
        events = list(handler.stream_events(self._make_request()))
        assert events[0].event_type == RESPONSE_CREATED

    def test_stream_last_event_is_response_completed(self):
        handler = ResponsesAPIHandler()
        events = list(handler.stream_events(self._make_request()))
        assert events[-1].event_type == RESPONSE_COMPLETED

    def test_stream_event_count_gte_three(self):
        handler = ResponsesAPIHandler()
        events = list(handler.stream_events(self._make_request()))
        assert len(events) >= 3

    def test_stream_completed_data_has_status(self):
        handler = ResponsesAPIHandler()
        events = list(handler.stream_events(self._make_request()))
        last = events[-1]
        assert last.data.get("status") == "completed"

    def test_stream_sequence_numbers_monotonic(self):
        handler = ResponsesAPIHandler()
        events = list(handler.stream_events(self._make_request()))
        seqs = [e.sequence_number for e in events]
        assert seqs == sorted(seqs)
