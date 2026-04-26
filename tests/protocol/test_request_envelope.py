"""Tests for request_envelope."""

from __future__ import annotations

import json

from src.protocol.request_envelope import (
    REQUEST_ENVELOPE_REGISTRY,
    EnvelopeSerializer,
    EnvelopeVersion,
    RequestEnvelope,
    ResponseEnvelope,
)


def test_request_default_id_length():
    r = RequestEnvelope()
    assert len(r.envelope_id) == 8


def test_request_default_version_v1():
    r = RequestEnvelope()
    assert r.version == EnvelopeVersion.V1


def test_request_default_path():
    assert RequestEnvelope().path == "/"


def test_request_default_method_empty():
    assert RequestEnvelope().method == ""


def test_request_default_headers_empty_dict():
    assert RequestEnvelope().headers == {}


def test_request_default_body_empty_bytes():
    assert RequestEnvelope().body == b""


def test_request_timestamp_present():
    r = RequestEnvelope()
    assert isinstance(r.timestamp_s, float)
    assert r.timestamp_s > 0


def test_request_is_frozen():
    r = RequestEnvelope()
    try:
        r.method = "POST"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen dataclass")


def test_response_defaults():
    resp = ResponseEnvelope(request_id="abc")
    assert resp.status_code == 200
    assert resp.body == b""
    assert resp.error == ""
    assert resp.latency_ms == 0.0


def test_response_is_frozen():
    resp = ResponseEnvelope(request_id="x")
    try:
        resp.status_code = 500  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen dataclass")


def test_encode_request_returns_bytes():
    r = RequestEnvelope(method="GET")
    assert isinstance(EnvelopeSerializer.encode_request(r), bytes)


def test_request_roundtrip_basic():
    r = RequestEnvelope(method="POST", path="/api", body=b"hello")
    blob = EnvelopeSerializer.encode_request(r)
    r2 = EnvelopeSerializer.decode_request(blob)
    assert r2.method == "POST"
    assert r2.path == "/api"
    assert r2.body == b"hello"


def test_request_roundtrip_preserves_id():
    r = RequestEnvelope()
    r2 = EnvelopeSerializer.decode_request(EnvelopeSerializer.encode_request(r))
    assert r2.envelope_id == r.envelope_id


def test_request_roundtrip_preserves_headers():
    r = RequestEnvelope(headers={"x-auth": "token"})
    r2 = EnvelopeSerializer.decode_request(EnvelopeSerializer.encode_request(r))
    assert r2.headers == {"x-auth": "token"}


def test_request_roundtrip_preserves_trace_id():
    r = RequestEnvelope(trace_id="trace-123")
    r2 = EnvelopeSerializer.decode_request(EnvelopeSerializer.encode_request(r))
    assert r2.trace_id == "trace-123"


def test_request_roundtrip_binary_body():
    body = bytes(range(256))
    r = RequestEnvelope(body=body)
    r2 = EnvelopeSerializer.decode_request(EnvelopeSerializer.encode_request(r))
    assert r2.body == body


def test_request_version_v2_roundtrip():
    r = RequestEnvelope(version=EnvelopeVersion.V2)
    r2 = EnvelopeSerializer.decode_request(EnvelopeSerializer.encode_request(r))
    assert r2.version == EnvelopeVersion.V2


def test_response_roundtrip():
    resp = ResponseEnvelope(request_id="r1", status_code=201, body=b"ok")
    blob = EnvelopeSerializer.encode_response(resp)
    r2 = EnvelopeSerializer.decode_response(blob)
    assert r2.request_id == "r1"
    assert r2.status_code == 201
    assert r2.body == b"ok"


def test_response_roundtrip_error():
    resp = ResponseEnvelope(request_id="r", status_code=500, error="boom")
    r2 = EnvelopeSerializer.decode_response(EnvelopeSerializer.encode_response(resp))
    assert r2.error == "boom"


def test_response_latency_roundtrip():
    resp = ResponseEnvelope(request_id="r", latency_ms=12.5)
    r2 = EnvelopeSerializer.decode_response(EnvelopeSerializer.encode_response(resp))
    assert r2.latency_ms == 12.5


def test_add_header_returns_new_envelope():
    r = RequestEnvelope()
    r2 = EnvelopeSerializer.add_header(r, "k", "v")
    assert r2 is not r
    assert r2.headers == {"k": "v"}


def test_add_header_does_not_mutate_original():
    r = RequestEnvelope()
    EnvelopeSerializer.add_header(r, "k", "v")
    assert r.headers == {}


def test_add_header_preserves_other_fields():
    r = RequestEnvelope(method="PUT", path="/p")
    r2 = EnvelopeSerializer.add_header(r, "a", "b")
    assert r2.method == "PUT"
    assert r2.path == "/p"


def test_add_header_overwrites_existing():
    r = RequestEnvelope(headers={"a": "1"})
    r2 = EnvelopeSerializer.add_header(r, "a", "2")
    assert r2.headers["a"] == "2"


def test_encode_uses_base64_for_body():
    r = RequestEnvelope(body=b"\x00\x01\x02")
    blob = EnvelopeSerializer.encode_request(r)
    obj = json.loads(blob.decode("utf-8"))
    assert "body_b64" in obj


def test_registry_default():
    assert REQUEST_ENVELOPE_REGISTRY["default"] is EnvelopeSerializer


def test_envelope_version_values():
    assert EnvelopeVersion.V1.value == "v1"
    assert EnvelopeVersion.V2.value == "v2"


def test_unique_envelope_ids():
    ids = {RequestEnvelope().envelope_id for _ in range(20)}
    assert len(ids) > 1
