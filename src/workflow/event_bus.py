import time
import uuid
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Event:
    name: str
    payload: dict
    source: str = ""
    timestamp_s: float = field(default_factory=time.monotonic)


@dataclass
class Subscription:
    sub_id: str
    event_pattern: str
    handler: Callable
    max_deliveries: int = -1
    delivered: int = 0


class EventBus:
    MAX_LOG = 1000

    def __init__(self) -> None:
        self._subs: dict[str, Subscription] = {}
        self._log: list[Event] = []

    def subscribe(
        self,
        event_pattern: str,
        handler: Callable,
        max_deliveries: int = -1,
    ) -> str:
        sub_id = uuid.uuid4().hex[:8]
        self._subs[sub_id] = Subscription(
            sub_id=sub_id,
            event_pattern=event_pattern,
            handler=handler,
            max_deliveries=max_deliveries,
        )
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        if sub_id in self._subs:
            del self._subs[sub_id]
            return True
        return False

    @staticmethod
    def _match(pattern: str, name: str) -> bool:
        if pattern == "*":
            return True
        return pattern == name

    def publish(self, event: Event) -> int:
        self._log.append(event)
        if len(self._log) > self.MAX_LOG:
            self._log = self._log[-self.MAX_LOG:]

        called = 0
        exhausted: list[str] = []
        # iterate over snapshot to allow safe mutation
        for sub_id, sub in list(self._subs.items()):
            if not self._match(sub.event_pattern, event.name):
                continue
            sub.handler(event)
            sub.delivered += 1
            called += 1
            if sub.max_deliveries > 0 and sub.delivered >= sub.max_deliveries:
                exhausted.append(sub_id)
        for sid in exhausted:
            self._subs.pop(sid, None)
        return called

    def publish_named(self, name: str, payload: dict, source: str = "") -> int:
        return self.publish(Event(name=name, payload=payload, source=source))

    def active_subscriptions(self) -> list[Subscription]:
        return list(self._subs.values())

    def event_log(self) -> list[Event]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()


EVENT_BUS_REGISTRY: dict[str, type[EventBus]] = {"default": EventBus}
