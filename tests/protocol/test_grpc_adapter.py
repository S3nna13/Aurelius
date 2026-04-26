"""Tests for grpc_adapter."""

from __future__ import annotations

from src.protocol.grpc_adapter import (
    GRPC_ADAPTER_REGISTRY,
    GRPCAdapter,
    GRPCMessage,
    GRPCResponse,
    GRPCStatus,
)


def test_build_request_service_method():
    msg = GRPCAdapter.build_request("Svc", "DoIt", {"a": 1})
    assert msg.service == "Svc"
    assert msg.method == "DoIt"


def test_build_request_payload():
    msg = GRPCAdapter.build_request("S", "M", {"x": 42})
    assert msg.payload == {"x": 42}


def test_build_request_metadata():
    msg = GRPCAdapter.build_request("S", "M", {}, auth="bearer")
    assert msg.metadata == {"auth": "bearer"}


def test_build_request_auto_message_id():
    msg = GRPCAdapter.build_request("S", "M", {})
    assert len(msg.message_id) == 8


def test_build_request_unique_ids():
    ids = {GRPCAdapter.build_request("S", "M", {}).message_id for _ in range(10)}
    assert len(ids) > 1


def test_build_response_default_ok():
    resp = GRPCAdapter.build_response("abc", {"r": 1})
    assert resp.status == GRPCStatus.OK
    assert resp.payload == {"r": 1}


def test_build_response_custom_status():
    resp = GRPCAdapter.build_response("x", {}, status=GRPCStatus.NOT_FOUND, error="nope")
    assert resp.status == GRPCStatus.NOT_FOUND
    assert resp.error_message == "nope"


def test_serialize_returns_bytes():
    msg = GRPCAdapter.build_request("S", "M", {"a": 1})
    assert isinstance(GRPCAdapter.serialize(msg), bytes)


def test_serialize_length_prefix_4_bytes():
    msg = GRPCAdapter.build_request("S", "M", {"a": 1})
    data = GRPCAdapter.serialize(msg)
    length = int.from_bytes(data[:4], "big")
    assert length == len(data) - 4


def test_roundtrip_preserves_service():
    msg = GRPCAdapter.build_request("MyService", "DoThing", {"k": "v"})
    data = GRPCAdapter.serialize(msg)
    decoded = GRPCAdapter.deserialize(data)
    assert decoded.service == "MyService"


def test_roundtrip_preserves_method():
    msg = GRPCAdapter.build_request("S", "Foo", {})
    decoded = GRPCAdapter.deserialize(GRPCAdapter.serialize(msg))
    assert decoded.method == "Foo"


def test_roundtrip_preserves_payload():
    payload = {"a": 1, "b": "two", "c": [1, 2, 3]}
    msg = GRPCAdapter.build_request("S", "M", payload)
    decoded = GRPCAdapter.deserialize(GRPCAdapter.serialize(msg))
    assert decoded.payload == payload


def test_roundtrip_preserves_message_id():
    msg = GRPCAdapter.build_request("S", "M", {})
    decoded = GRPCAdapter.deserialize(GRPCAdapter.serialize(msg))
    assert decoded.message_id == msg.message_id


def test_roundtrip_preserves_metadata():
    msg = GRPCAdapter.build_request("S", "M", {}, token="abc", user="u1")  # noqa: S106
    decoded = GRPCAdapter.deserialize(GRPCAdapter.serialize(msg))
    assert decoded.metadata == {"token": "abc", "user": "u1"}


def test_deserialize_too_short_raises():
    import pytest

    with pytest.raises(ValueError):
        GRPCAdapter.deserialize(b"\x00")


def test_invoke_success():
    msg = GRPCAdapter.build_request("S", "M", {"x": 5})
    resp = GRPCAdapter.invoke(msg, lambda p: {"result": p["x"] * 2})
    assert resp.status == GRPCStatus.OK
    assert resp.payload == {"result": 10}


def test_invoke_preserves_message_id():
    msg = GRPCAdapter.build_request("S", "M", {})
    resp = GRPCAdapter.invoke(msg, lambda p: {})
    assert resp.message_id == msg.message_id


def test_invoke_exception_returns_internal():
    msg = GRPCAdapter.build_request("S", "M", {})

    def boom(_p):
        raise RuntimeError("fail")

    resp = GRPCAdapter.invoke(msg, boom)
    assert resp.status == GRPCStatus.INTERNAL
    assert "fail" in resp.error_message


def test_invoke_exception_empty_payload():
    msg = GRPCAdapter.build_request("S", "M", {})
    resp = GRPCAdapter.invoke(msg, lambda p: (_ for _ in ()).throw(ValueError("x")))
    assert resp.payload == {}


def test_invoke_non_dict_result_is_internal():
    msg = GRPCAdapter.build_request("S", "M", {})
    resp = GRPCAdapter.invoke(msg, lambda p: "not a dict")  # type: ignore[return-value]
    assert resp.status == GRPCStatus.INTERNAL


def test_is_success_true_for_ok():
    resp = GRPCAdapter.build_response("x", {}, status=GRPCStatus.OK)
    assert GRPCAdapter.is_success(resp) is True


def test_is_success_false_for_internal():
    resp = GRPCAdapter.build_response("x", {}, status=GRPCStatus.INTERNAL)
    assert GRPCAdapter.is_success(resp) is False


def test_status_values():
    assert GRPCStatus.OK == 0
    assert GRPCStatus.CANCELLED == 1
    assert GRPCStatus.INVALID_ARGUMENT == 3
    assert GRPCStatus.NOT_FOUND == 5
    assert GRPCStatus.ALREADY_EXISTS == 6
    assert GRPCStatus.INTERNAL == 13
    assert GRPCStatus.UNAVAILABLE == 14


def test_grpc_message_frozen():
    msg = GRPCMessage(service="s", method="m", payload={})
    try:
        msg.service = "x"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen")


def test_grpc_response_frozen():
    resp = GRPCResponse(message_id="x", status=GRPCStatus.OK, payload={})
    try:
        resp.status = GRPCStatus.INTERNAL  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen")


def test_registry_default():
    assert GRPC_ADAPTER_REGISTRY["default"] is GRPCAdapter


def test_build_request_copies_payload():
    original = {"x": 1}
    msg = GRPCAdapter.build_request("S", "M", original)
    original["x"] = 99
    assert msg.payload["x"] == 1
