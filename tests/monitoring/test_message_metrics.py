"""Tests for message_metrics — instrumentation wrapper for MessageBus."""

from __future__ import annotations

import pytest

from src.monitoring.message_metrics import (
    DEFAULT_MESSAGE_METRICS,
    MESSAGE_METRICS_REGISTRY,
    MessageMetrics,
    MessageMetricsConfig,
)
from src.monitoring.prometheus_metrics import MetricsCollector
from src.multiagent.message_bus import AgentMessage, MessageBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_collector() -> MetricsCollector:
    return MetricsCollector()


def make_bus() -> MessageBus:
    return MessageBus()


# ---------------------------------------------------------------------------
# Record sent
# ---------------------------------------------------------------------------


def test_record_sent_increments_counter():
    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    mm.record_sent(msg_type="alert", latency_seconds=0.01)
    assert (
        col.read_counter("aurelius_message_bus_messages_sent_total", {"msg_type": "alert"}) == 1.0
    )


def test_record_sent_records_histogram():
    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    mm.record_sent(latency_seconds=0.05)
    hist = col.read_histogram("aurelius_message_bus_send_latency_seconds", {"msg_type": "unknown"})
    assert hist == [0.05]


def test_record_sent_error_increments_error_counter():
    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    mm.record_sent(error=True)
    assert (
        col.read_counter("aurelius_message_bus_send_errors_total", {"msg_type": "unknown"}) == 1.0
    )


# ---------------------------------------------------------------------------
# Record received
# ---------------------------------------------------------------------------


def test_record_received_increments_counter():
    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    mm.record_received(msg_type="alert", count=3)
    assert (
        col.read_counter("aurelius_message_bus_messages_received_total", {"msg_type": "alert"})
        == 3.0
    )


# ---------------------------------------------------------------------------
# Record pending
# ---------------------------------------------------------------------------


def test_record_pending_sets_gauge():
    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    mm.record_pending(7)
    assert col.read_gauge("aurelius_message_bus_pending_messages") == 7.0


# ---------------------------------------------------------------------------
# InstrumentedMessageBus send
# ---------------------------------------------------------------------------


def test_instrumented_send_records_metrics():
    col = make_collector()
    bus = make_bus()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    wrapped = mm.wrap(bus)
    msg = AgentMessage("alice", "bob", "greet", "hello")
    wrapped.send(msg)
    assert (
        col.read_counter("aurelius_message_bus_messages_sent_total", {"msg_type": "greet"}) == 1.0
    )


def test_instrumented_send_records_error_on_exception():
    class BrokenBus(MessageBus):
        def send(self, message: AgentMessage) -> None:
            raise RuntimeError("boom")

    col = make_collector()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    wrapped = mm.wrap(BrokenBus())
    with pytest.raises(RuntimeError):
        wrapped.send(AgentMessage("alice", "bob", "greet", "hello"))
    assert col.read_counter("aurelius_message_bus_send_errors_total", {"msg_type": "greet"}) == 1.0


# ---------------------------------------------------------------------------
# InstrumentedMessageBus receive
# ---------------------------------------------------------------------------


def test_instrumented_receive_records_metrics():
    col = make_collector()
    bus = make_bus()
    bus.send(AgentMessage("alice", "bob", "alert", "ping"))
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    wrapped = mm.wrap(bus)
    msgs = wrapped.receive("bob")
    assert len(msgs) == 1
    assert (
        col.read_counter("aurelius_message_bus_messages_received_total", {"msg_type": "alert"})
        == 1.0
    )


def test_instrumented_receive_no_metrics_when_empty():
    col = make_collector()
    bus = make_bus()
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    wrapped = mm.wrap(bus)
    wrapped.receive("nobody")
    assert col.read_counter("aurelius_message_bus_messages_received_total") == 0.0


# ---------------------------------------------------------------------------
# InstrumentedMessageBus pending_count
# ---------------------------------------------------------------------------


def test_instrumented_pending_sets_gauge():
    col = make_collector()
    bus = make_bus()
    bus.send(AgentMessage("alice", "bob", "alert", "ping"))
    mm = MessageMetrics(MessageMetricsConfig(collector=col))
    wrapped = mm.wrap(bus)
    count = wrapped.pending_count("bob")
    assert count == 1
    assert col.read_gauge("aurelius_message_bus_pending_messages") == 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in MESSAGE_METRICS_REGISTRY
    assert isinstance(MESSAGE_METRICS_REGISTRY["default"], MessageMetrics)


def test_default_is_message_metrics():
    assert isinstance(DEFAULT_MESSAGE_METRICS, MessageMetrics)
