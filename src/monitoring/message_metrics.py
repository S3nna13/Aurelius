"""Message bus metrics integration for the monitoring surface.

Wraps send/receive operations on a MessageBus and emits counters,
gauges, and histograms via MetricsCollector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.monitoring.prometheus_metrics import MetricsCollector, METRICS_COLLECTOR
from src.multiagent.message_bus import AgentMessage, MessageBus


@dataclass
class MessageMetricsConfig:
    collector: MetricsCollector = METRICS_COLLECTOR
    prefix: str = "aurelius_message_bus"
    latency_buckets: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)


class MessageMetrics:
    """Instrumentation wrapper for MessageBus operations.

    Usage:
        bus = MessageBus()
        metrics = MessageMetrics()
        instrumented = metrics.wrap(bus)
        instrumented.send(msg)   # emits counter + histogram
    """

    def __init__(self, config: MessageMetricsConfig | None = None) -> None:
        self._config = config or MessageMetricsConfig()
        self._collector = self._config.collector

    def _name(self, metric: str) -> str:
        return f"{self._config.prefix}_{metric}"

    def record_sent(
        self,
        msg_type: str = "unknown",
        latency_seconds: float = 0.0,
        error: bool = False,
    ) -> None:
        labels = {"msg_type": msg_type}
        self._collector.increment_counter(
            self._name("messages_sent_total"), 1.0, labels
        )
        self._collector.observe_histogram(
            self._name("send_latency_seconds"), latency_seconds, labels
        )
        if error:
            self._collector.increment_counter(
                self._name("send_errors_total"), 1.0, labels
            )

    def record_received(self, msg_type: str = "unknown", count: int = 1) -> None:
        labels = {"msg_type": msg_type}
        self._collector.increment_counter(
            self._name("messages_received_total"), float(count), labels
        )

    def record_pending(self, count: int) -> None:
        self._collector.set_gauge(self._name("pending_messages"), float(count))

    def wrap(self, bus: MessageBus) -> "InstrumentedMessageBus":
        """Return an InstrumentedMessageBus that proxies to *bus*."""
        return InstrumentedMessageBus(bus, self)


class InstrumentedMessageBus:
    """Proxy for MessageBus that records metrics on every operation."""

    def __init__(self, bus: MessageBus, metrics: MessageMetrics) -> None:
        self._bus = bus
        self._metrics = metrics

    def send(self, message: AgentMessage) -> None:
        start = time.perf_counter()
        error = False
        try:
            self._bus.send(message)
        except Exception:
            error = True
            raise
        finally:
            latency = time.perf_counter() - start
            self._metrics.record_sent(
                msg_type=message.msg_type,
                latency_seconds=latency,
                error=error,
            )

    def receive(self, agent_id: str) -> list[AgentMessage]:
        msgs = self._bus.receive(agent_id)
        if msgs:
            # approximate: count by msg_type of first message for label
            self._metrics.record_received(
                msg_type=msgs[0].msg_type, count=len(msgs)
            )
        return msgs

    def pending_count(self, agent_id: str | None = None) -> int:
        count = self._bus.pending_count(agent_id)
        self._metrics.record_pending(count)
        return count


# Module-level registry
MESSAGE_METRICS_REGISTRY: dict[str, MessageMetrics] = {}
DEFAULT_MESSAGE_METRICS = MessageMetrics()
MESSAGE_METRICS_REGISTRY["default"] = DEFAULT_MESSAGE_METRICS
