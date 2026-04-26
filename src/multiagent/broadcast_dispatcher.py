"""Broadcast dispatcher for multi-agent message bus.

Extends the simple MessageBus with topic-based broadcast and
subscriber-group routing. All inputs treated as untrusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.multiagent.message_bus import AgentMessage, MessageBus


@dataclass
class BroadcastDispatcher:
    """Wraps a MessageBus with topic-based broadcast dispatch.

    Agents subscribe to topics. Publishing to a topic delivers a copy
    of the message to every subscriber's inbox.
    """

    _bus: MessageBus = field(default_factory=MessageBus)
    _subscriptions: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Register *agent_id* as a listener on *topic*."""
        self._subscriptions.setdefault(topic, set()).add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Remove *agent_id* from *topic*. Returns True if it was present."""
        subs = self._subscriptions.get(topic, set())
        if agent_id in subs:
            subs.discard(agent_id)
            if not subs:
                del self._subscriptions[topic]
            return True
        return False

    def publish(self, topic: str, payload: Any, sender: str = "system") -> int:
        """Broadcast a message on *topic* to all subscribers.

        Returns the number of recipients that received the message.
        """
        recipients = list(self._subscriptions.get(topic, set()))
        for recipient in recipients:
            msg = AgentMessage(
                sender=sender,
                recipient=recipient,
                msg_type=topic,
                payload=payload,
            )
            self._bus.send(msg)
        return len(recipients)

    def subscribers(self, topic: str) -> list[str]:
        """Return a snapshot of current subscribers for *topic*."""
        return list(self._subscriptions.get(topic, set()))

    def topics(self) -> list[str]:
        """Return all topics with at least one subscriber."""
        return list(self._subscriptions.keys())

    def pending_count(self, agent_id: str | None = None) -> int:
        """Delegate pending count to the underlying bus."""
        return self._bus.pending_count(agent_id)

    def receive(self, agent_id: str) -> list[AgentMessage]:
        """Delegate receive to the underlying bus."""
        return self._bus.receive(agent_id)


# Module-level registry
BROADCAST_DISPATCHER_REGISTRY: dict[str, BroadcastDispatcher] = {}
DEFAULT_BROADCAST_DISPATCHER = BroadcastDispatcher()
BROADCAST_DISPATCHER_REGISTRY["default"] = DEFAULT_BROADCAST_DISPATCHER
