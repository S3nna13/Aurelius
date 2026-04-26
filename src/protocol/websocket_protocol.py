"""WebSocket protocol handler: frame model and connection state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, StrEnum


class WSOpcode(int, Enum):
    CONTINUATION = 0
    TEXT = 1
    BINARY = 2
    CLOSE = 8
    PING = 9
    PONG = 10


@dataclass(frozen=True)
class WSFrame:
    opcode: WSOpcode
    payload: bytes
    fin: bool = True
    masked: bool = False


class WSConnectionState(StrEnum):
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class WebSocketProtocol:
    def __init__(self) -> None:
        self.state: WSConnectionState = WSConnectionState.CONNECTING
        self.rx_buffer: list[WSFrame] = []
        self.tx_queue: deque[WSFrame] = deque()

    def on_open(self) -> list[WSFrame]:
        self.state = WSConnectionState.OPEN
        return []

    def on_message(self, data: bytes, binary: bool = False) -> WSFrame:
        opcode = WSOpcode.BINARY if binary else WSOpcode.TEXT
        return WSFrame(opcode=opcode, payload=data)

    def on_ping(self, payload: bytes = b"") -> WSFrame:
        return WSFrame(opcode=WSOpcode.PONG, payload=payload)

    def on_close(self, code: int = 1000, reason: str = "Normal closure") -> WSFrame:
        payload = code.to_bytes(2, "big") + reason.encode("utf-8")
        return WSFrame(opcode=WSOpcode.CLOSE, payload=payload)

    def send_text(self, text: str) -> WSFrame:
        frame = WSFrame(opcode=WSOpcode.TEXT, payload=text.encode("utf-8"))
        self.tx_queue.append(frame)
        return frame

    def send_binary(self, data: bytes) -> WSFrame:
        frame = WSFrame(opcode=WSOpcode.BINARY, payload=data)
        self.tx_queue.append(frame)
        return frame

    def receive(self, frame: WSFrame) -> None:
        self.rx_buffer.append(frame)
        if frame.opcode == WSOpcode.CLOSE:
            self.state = WSConnectionState.CLOSING

    def pending_frames(self) -> list[WSFrame]:
        drained = list(self.tx_queue)
        self.tx_queue.clear()
        return drained

    def is_open(self) -> bool:
        return self.state == WSConnectionState.OPEN


WEBSOCKET_PROTOCOL_REGISTRY = {"default": WebSocketProtocol}
