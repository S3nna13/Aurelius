"""Tests for websocket_protocol."""

from __future__ import annotations

from src.protocol.websocket_protocol import (
    WEBSOCKET_PROTOCOL_REGISTRY,
    WebSocketProtocol,
    WSConnectionState,
    WSFrame,
    WSOpcode,
)


def test_initial_state_connecting():
    assert WebSocketProtocol().state == WSConnectionState.CONNECTING


def test_on_open_transitions_to_open():
    ws = WebSocketProtocol()
    frames = ws.on_open()
    assert ws.state == WSConnectionState.OPEN
    assert frames == []


def test_is_open_after_open():
    ws = WebSocketProtocol()
    ws.on_open()
    assert ws.is_open()


def test_is_open_false_initially():
    assert WebSocketProtocol().is_open() is False


def test_on_message_text():
    ws = WebSocketProtocol()
    frame = ws.on_message(b"hello", binary=False)
    assert frame.opcode == WSOpcode.TEXT
    assert frame.payload == b"hello"


def test_on_message_binary():
    ws = WebSocketProtocol()
    frame = ws.on_message(b"\x00\x01", binary=True)
    assert frame.opcode == WSOpcode.BINARY


def test_on_ping_returns_pong():
    ws = WebSocketProtocol()
    frame = ws.on_ping(b"ping-data")
    assert frame.opcode == WSOpcode.PONG
    assert frame.payload == b"ping-data"


def test_on_ping_empty_payload():
    ws = WebSocketProtocol()
    frame = ws.on_ping()
    assert frame.opcode == WSOpcode.PONG
    assert frame.payload == b""


def test_on_close_default_code():
    ws = WebSocketProtocol()
    frame = ws.on_close()
    assert frame.opcode == WSOpcode.CLOSE
    code = int.from_bytes(frame.payload[:2], "big")
    assert code == 1000


def test_on_close_custom_code_and_reason():
    ws = WebSocketProtocol()
    frame = ws.on_close(code=1002, reason="bye")
    code = int.from_bytes(frame.payload[:2], "big")
    assert code == 1002
    assert frame.payload[2:] == b"bye"


def test_send_text_enqueues():
    ws = WebSocketProtocol()
    ws.send_text("hi")
    assert len(ws.tx_queue) == 1


def test_send_text_returns_text_frame():
    ws = WebSocketProtocol()
    frame = ws.send_text("abc")
    assert frame.opcode == WSOpcode.TEXT
    assert frame.payload == b"abc"


def test_send_binary_enqueues():
    ws = WebSocketProtocol()
    ws.send_binary(b"\xff")
    assert len(ws.tx_queue) == 1


def test_send_binary_frame_type():
    ws = WebSocketProtocol()
    frame = ws.send_binary(b"\x00")
    assert frame.opcode == WSOpcode.BINARY


def test_receive_appends_to_rx_buffer():
    ws = WebSocketProtocol()
    frame = WSFrame(opcode=WSOpcode.TEXT, payload=b"msg")
    ws.receive(frame)
    assert ws.rx_buffer == [frame]


def test_receive_close_transitions_to_closing():
    ws = WebSocketProtocol()
    ws.on_open()
    ws.receive(WSFrame(opcode=WSOpcode.CLOSE, payload=b"\x03\xe8"))
    assert ws.state == WSConnectionState.CLOSING


def test_receive_text_does_not_change_state():
    ws = WebSocketProtocol()
    ws.on_open()
    ws.receive(WSFrame(opcode=WSOpcode.TEXT, payload=b"x"))
    assert ws.state == WSConnectionState.OPEN


def test_pending_frames_drains_queue():
    ws = WebSocketProtocol()
    ws.send_text("a")
    ws.send_text("b")
    pending = ws.pending_frames()
    assert len(pending) == 2
    assert ws.pending_frames() == []


def test_pending_frames_order():
    ws = WebSocketProtocol()
    ws.send_text("first")
    ws.send_text("second")
    pending = ws.pending_frames()
    assert pending[0].payload == b"first"
    assert pending[1].payload == b"second"


def test_ws_frame_is_frozen():
    frame = WSFrame(opcode=WSOpcode.TEXT, payload=b"x")
    try:
        frame.payload = b"y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen")


def test_ws_frame_default_fin_true():
    frame = WSFrame(opcode=WSOpcode.TEXT, payload=b"x")
    assert frame.fin is True


def test_ws_frame_default_masked_false():
    frame = WSFrame(opcode=WSOpcode.TEXT, payload=b"x")
    assert frame.masked is False


def test_opcode_values():
    assert WSOpcode.TEXT == 1
    assert WSOpcode.BINARY == 2
    assert WSOpcode.CLOSE == 8
    assert WSOpcode.PING == 9
    assert WSOpcode.PONG == 10


def test_state_enum_members():
    assert WSConnectionState.CONNECTING.value == "connecting"
    assert WSConnectionState.OPEN.value == "open"
    assert WSConnectionState.CLOSING.value == "closing"
    assert WSConnectionState.CLOSED.value == "closed"


def test_registry_default():
    assert WEBSOCKET_PROTOCOL_REGISTRY["default"] is WebSocketProtocol


def test_multiple_receives_accumulate():
    ws = WebSocketProtocol()
    for i in range(5):
        ws.receive(WSFrame(opcode=WSOpcode.TEXT, payload=bytes([i])))
    assert len(ws.rx_buffer) == 5


def test_on_close_payload_length():
    ws = WebSocketProtocol()
    frame = ws.on_close(code=1000, reason="ok")
    assert len(frame.payload) == 2 + len("ok")
