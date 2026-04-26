"""Request/response envelope wrapper with JSON + base64 serialization."""

from __future__ import annotations

import base64
import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class EnvelopeVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"


def _hex8() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class RequestEnvelope:
    envelope_id: str = field(default_factory=_hex8)
    version: EnvelopeVersion = EnvelopeVersion.V1
    method: str = ""
    path: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    timestamp_s: float = field(default_factory=time.monotonic)
    trace_id: str = ""


@dataclass(frozen=True)
class ResponseEnvelope:
    request_id: str
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    latency_ms: float = 0.0
    error: str = ""


class EnvelopeSerializer:
    @staticmethod
    def encode_request(req: RequestEnvelope) -> bytes:
        obj = {
            "envelope_id": req.envelope_id,
            "version": req.version.value,
            "method": req.method,
            "path": req.path,
            "headers": req.headers,
            "body_b64": base64.b64encode(req.body).decode("ascii"),
            "timestamp_s": req.timestamp_s,
            "trace_id": req.trace_id,
        }
        return json.dumps(obj).encode("utf-8")

    @staticmethod
    def decode_request(data: bytes) -> RequestEnvelope:
        obj = json.loads(data.decode("utf-8"))
        return RequestEnvelope(
            envelope_id=obj["envelope_id"],
            version=EnvelopeVersion(obj["version"]),
            method=obj["method"],
            path=obj["path"],
            headers=dict(obj["headers"]),
            body=base64.b64decode(obj["body_b64"].encode("ascii")),
            timestamp_s=obj["timestamp_s"],
            trace_id=obj["trace_id"],
        )

    @staticmethod
    def encode_response(resp: ResponseEnvelope) -> bytes:
        obj = {
            "request_id": resp.request_id,
            "status_code": resp.status_code,
            "headers": resp.headers,
            "body_b64": base64.b64encode(resp.body).decode("ascii"),
            "latency_ms": resp.latency_ms,
            "error": resp.error,
        }
        return json.dumps(obj).encode("utf-8")

    @staticmethod
    def decode_response(data: bytes) -> ResponseEnvelope:
        obj = json.loads(data.decode("utf-8"))
        return ResponseEnvelope(
            request_id=obj["request_id"],
            status_code=obj["status_code"],
            headers=dict(obj["headers"]),
            body=base64.b64decode(obj["body_b64"].encode("ascii")),
            latency_ms=obj["latency_ms"],
            error=obj["error"],
        )

    @staticmethod
    def add_header(req: RequestEnvelope, key: str, value: str) -> RequestEnvelope:
        new_headers = dict(req.headers)
        new_headers[key] = value
        return dataclasses.replace(req, headers=new_headers)


REQUEST_ENVELOPE_REGISTRY = {"default": EnvelopeSerializer}
