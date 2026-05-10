"""Hermes notification system for Aurelius.

Routes agent events, system alerts, and user messages to configurable
delivery channels.  Core channels: web-push (SSE), webhook.  SMS and
email are预留 as future channels.

Intended to run on a home network with the ability to notify the
owner's phone in the future via SMS gateways or push services.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class HermesError(Exception):
    """Raised for Hermes configuration or delivery errors."""


@dataclass
class Notification:
    """A single notification payload."""

    notification_id: str
    timestamp: float
    channel: str  # "web", "sms", "email", "webhook"
    priority: str  # "critical", "high", "normal", "low"
    category: str  # "agent", "system", "alert", "user"
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    read: bool = False
    delivered: bool = False


@dataclass
class DeliveryResult:
    """Result of a delivery attempt."""

    success: bool
    channel: str
    error: str | None = None


class HermesNotifier:
    """Notification router with in-memory queue and channel dispatch.

    Parameters
    ----------
    max_queue_size:
        Hard cap on undelivered notifications to prevent memory
        exhaustion (default 10_000).
    ttl_seconds:
        Age at which delivered notifications are eligible for
        eviction during cleanup (default 86_400 = 24 h).
    """

    def __init__(
        self,
        max_queue_size: int = 10_000,
        ttl_seconds: float = 86_400.0,
    ) -> None:
        self._max_queue_size = max_queue_size
        self._ttl = ttl_seconds
        self._queue: list[Notification] = []
        self._listeners: list[Callable[[Notification], None]] = []
        self._lock = threading.Lock()
        self._webhook_url: str | None = None
        self._webhook_headers: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_webhook(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Set the webhook URL for the ``webhook`` channel."""
        self._webhook_url = url
        self._webhook_headers = dict(headers) if headers else {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def notify(
        self,
        title: str,
        body: str,
        *,
        channel: str = "web",
        priority: str = "normal",
        category: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> Notification:
        """Enqueue a notification and attempt immediate delivery."""
        notification = Notification(
            notification_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            channel=channel,
            priority=priority,
            category=category,
            title=title,
            body=body,
            metadata=dict(metadata) if metadata else {},
        )

        with self._lock:
            if len(self._queue) >= self._max_queue_size:
                # Evict oldest delivered item, or oldest overall if none delivered
                evict_idx = next(
                    (
                        i
                        for i, n in enumerate(self._queue)
                        if n.delivered and (time.time() - n.timestamp) > self._ttl
                    ),
                    0,
                )
                self._queue.pop(evict_idx)
            self._queue.append(notification)

        # Attempt delivery
        result = self._deliver(notification)
        if result.success:
            notification.delivered = True

        # Fan-out to in-memory listeners (for SSE push)
        for listener in self._listeners:
            try:
                listener(notification)
            except Exception:
                logger.exception("Hermes listener error")

        return notification

    # ------------------------------------------------------------------
    # Query & lifecycle
    # ------------------------------------------------------------------

    def list_notifications(
        self,
        *,
        category: str | None = None,
        priority: str | None = None,
        read: bool | None = None,
        limit: int = 100,
    ) -> list[Notification]:
        """Return matching notifications, newest first."""
        with self._lock:
            filtered = list(self._queue)
        filtered.reverse()

        if category is not None:
            filtered = [n for n in filtered if n.category == category]
        if priority is not None:
            filtered = [n for n in filtered if n.priority == priority]
        if read is not None:
            filtered = [n for n in filtered if n.read == read]

        return filtered[:limit]

    def mark_read(self, notification_id: str) -> bool:
        """Mark a single notification as read."""
        with self._lock:
            for n in self._queue:
                if n.notification_id == notification_id:
                    n.read = True
                    return True
        return False

    def mark_all_read(self, category: str | None = None) -> int:
        """Mark notifications as read. Returns count changed."""
        changed = 0
        with self._lock:
            for n in self._queue:
                if category is None or n.category == category:
                    if not n.read:
                        n.read = True
                        changed += 1
        return changed

    def unread_count(self, category: str | None = None) -> int:
        """Return number of unread notifications."""
        with self._lock:
            return sum(
                1
                for n in self._queue
                if not n.read and (category is None or n.category == category)
            )

    def clear(self) -> None:
        """Drop all notifications."""
        with self._lock:
            self._queue.clear()

    # ------------------------------------------------------------------
    # Listeners (for SSE / WebSocket fan-out)
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[Notification], None]) -> Callable[[], None]:
        """Subscribe to new notifications. Returns an unsubscribe function."""
        self._listeners.append(callback)

        def _unsub() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return _unsub

    # ------------------------------------------------------------------
    # Delivery implementations
    # ------------------------------------------------------------------

    def _deliver(self, notification: Notification) -> DeliveryResult:
        if notification.channel == "web":
            # Web is fire-and-forget via listeners
            return DeliveryResult(success=True, channel="web")

        if notification.channel == "webhook":
            return self._deliver_webhook(notification)

        if notification.channel == "sms":
            # Placeholder for future Twilio / SNS integration
            logger.info("Hermes SMS placeholder: %s — %s", notification.title, notification.body)
            return DeliveryResult(
                success=True, channel="sms", error="SMS delivery not yet implemented"
            )

        if notification.channel == "email":
            logger.info("Hermes email placeholder: %s", notification.title)
            return DeliveryResult(
                success=True,
                channel="email",
                error="Email delivery not yet implemented",
            )

        return DeliveryResult(success=False, channel=notification.channel, error="Unknown channel")

    def _deliver_webhook(self, notification: Notification) -> DeliveryResult:
        if self._webhook_url is None:
            return DeliveryResult(
                success=False, channel="webhook", error="Webhook URL not configured"
            )
        try:
            req = Request(  # noqa: S310
                self._webhook_url,
                data=json.dumps(
                    {
                        "id": notification.notification_id,
                        "timestamp": notification.timestamp,
                        "priority": notification.priority,
                        "category": notification.category,
                        "title": notification.title,
                        "body": notification.body,
                        "metadata": notification.metadata,
                    }
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **self._webhook_headers,
                },
            )
            with urlopen(req, timeout=10) as resp:  # noqa: S310
                if resp.status < 300:
                    return DeliveryResult(success=True, channel="webhook")
                return DeliveryResult(
                    success=False,
                    channel="webhook",
                    error=f"HTTP {resp.status}",
                )
        except Exception as exc:
            return DeliveryResult(success=False, channel="webhook", error=str(exc))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return queue statistics."""
        with self._lock:
            total = len(self._queue)
            unread = sum(1 for n in self._queue if not n.read)
            by_channel: dict[str, int] = {}
            by_priority: dict[str, int] = {}
            for n in self._queue:
                by_channel[n.channel] = by_channel.get(n.channel, 0) + 1
                by_priority[n.priority] = by_priority.get(n.priority, 0) + 1
        return {
            "total": total,
            "unread": unread,
            "by_channel": by_channel,
            "by_priority": by_priority,
            "max_queue_size": self._max_queue_size,
        }


# Backfill the Callable import used by subscribe
from collections.abc import Callable  # noqa: E402
from urllib.request import Request, urlopen  # noqa: E402

# ---------------------------------------------------------------------------
# Singleton & registry
# ---------------------------------------------------------------------------

DEFAULT_HERMES_NOTIFIER = HermesNotifier()

HERMES_NOTIFIER_REGISTRY: dict[str, HermesNotifier] = {
    "default": DEFAULT_HERMES_NOTIFIER,
}
