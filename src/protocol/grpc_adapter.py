"""gRPC-style message format adapter (pure Python, no grpcio)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class GRPCStatus(int, Enum):
    OK = 0
    CANCELLED = 1
    INVALID_ARGUMENT = 3
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    INTERNAL = 13
    UNAVAILABLE = 14


def _hex8() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class GRPCMessage:
    service: str
    method: str
    payload: dict
    message_id: str = field(default_factory=_hex8)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GRPCResponse:
    message_id: str
    status: GRPCStatus
    payload: dict
    error_message: str = ""
    trailing_metadata: dict[str, str] = field(default_factory=dict)


class GRPCAdapter:
    @staticmethod
    def build_request(service: str, method: str, payload: dict, **metadata: str) -> GRPCMessage:
        return GRPCMessage(
            service=service,
            method=method,
            payload=dict(payload),
            metadata=dict(metadata),
        )

    @staticmethod
    def build_response(
        message_id: str,
        payload: dict,
        status: GRPCStatus = GRPCStatus.OK,
        error: str = "",
    ) -> GRPCResponse:
        return GRPCResponse(
            message_id=message_id,
            status=status,
            payload=dict(payload),
            error_message=error,
        )

    @staticmethod
    def serialize(msg: GRPCMessage) -> bytes:
        obj = {
            "service": msg.service,
            "method": msg.method,
            "payload": msg.payload,
            "message_id": msg.message_id,
            "metadata": msg.metadata,
        }
        body = json.dumps(obj).encode("utf-8")
        length = len(body).to_bytes(4, "big")
        return length + body

    @staticmethod
    def deserialize(data: bytes) -> GRPCMessage:
        if len(data) < 4:
            raise ValueError("grpc frame too short")
        length = int.from_bytes(data[:4], "big")
        body = data[4 : 4 + length]
        obj = json.loads(body.decode("utf-8"))
        return GRPCMessage(
            service=obj["service"],
            method=obj["method"],
            payload=dict(obj["payload"]),
            message_id=obj["message_id"],
            metadata=dict(obj["metadata"]),
        )

    @staticmethod
    def invoke(msg: GRPCMessage, handler: Callable[[dict], dict]) -> GRPCResponse:
        try:
            result = handler(msg.payload)
            if not isinstance(result, dict):
                raise TypeError("handler must return dict")
            return GRPCResponse(
                message_id=msg.message_id,
                status=GRPCStatus.OK,
                payload=result,
            )
        except Exception as exc:  # noqa: BLE001
            return GRPCResponse(
                message_id=msg.message_id,
                status=GRPCStatus.INTERNAL,
                payload={},
                error_message=str(exc),
            )

    @staticmethod
    def is_success(resp: GRPCResponse) -> bool:
        return resp.status == GRPCStatus.OK


GRPC_ADAPTER_REGISTRY = {"default": GRPCAdapter}
