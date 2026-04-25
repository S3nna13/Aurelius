"""Agent message bus: pub/sub, routing, envelope stamping."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _hex8() -> str:
    return uuid.uuid4().hex[:8]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MessageEnvelope:
    sender: str
    recipient: str
    topic: str
    payload: dict
    id: str = field(default_factory=_hex8)
    timestamp: str = field(default_factory=_utcnow_iso)
    priority: int = 0


@dataclass
class AgentMessage:
    envelope: MessageEnvelope
    ack_required: bool = False
    ttl_seconds: float = 60.0


class MessageBus:
    def __init__(self) -> None:
        self._subscriptions: dict[str, list[str]] = {}
        self._mailboxes: dict[str, list[AgentMessage]] = {}

    def subscribe(self, topic: str, subscriber_id: str) -> None:
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        if subscriber_id not in self._subscriptions[topic]:
            self._subscriptions[topic].append(subscriber_id)
        if subscriber_id not in self._mailboxes:
            self._mailboxes[subscriber_id] = []

    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        subs = self._subscriptions.get(topic, [])
        if subscriber_id in subs:
            subs.remove(subscriber_id)
            return True
        return False

    def publish(self, msg: AgentMessage) -> list[str]:
        topic = msg.envelope.topic
        subscribers = list(self._subscriptions.get(topic, []))
        for sub_id in subscribers:
            if sub_id not in self._mailboxes:
                self._mailboxes[sub_id] = []
            self._mailboxes[sub_id].append(msg)
        return subscribers

    def receive(self, subscriber_id: str) -> list[AgentMessage]:
        messages = list(self._mailboxes.get(subscriber_id, []))
        self._mailboxes[subscriber_id] = []
        return messages

    def subscribers(self, topic: str) -> list[str]:
        return list(self._subscriptions.get(topic, []))

    def pending_count(self, subscriber_id: str) -> int:
        return len(self._mailboxes.get(subscriber_id, []))


MESSAGE_BUS = MessageBus()
