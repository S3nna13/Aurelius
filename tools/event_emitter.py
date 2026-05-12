"""Event emitter for publish/subscribe communication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    name: str
    data: dict[str, Any] | None = None


@dataclass
class EventEmitter:
    """Simple pub/sub event emitter for tool/agent communication."""

    _listeners: dict[str, list[Callable]] = field(default_factory=dict, repr=False)

    def on(self, event_name: str, handler: Callable) -> None:
        self._listeners.setdefault(event_name, []).append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._listeners.get(event.name, []):
            handler(event)

    def off(self, event_name: str, handler: Callable) -> None:
        handlers = self._listeners.get(event_name, [])
        if handler in handlers:
            handlers.remove(handler)

    def listener_count(self, event_name: str) -> int:
        return len(self._listeners.get(event_name, []))


EVENT_EMITTER = EventEmitter()
