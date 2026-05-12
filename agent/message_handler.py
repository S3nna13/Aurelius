"""Message handler for agent inbox processing.

Routes inbound AgentMessage objects to registered handlers by msg_type.
Fail closed: unregistered types are rejected with a loud error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.multiagent.message_bus import AgentMessage

Handler = Callable[[AgentMessage], Any]


@dataclass
class MessageHandler:
    """Stateful router for agent-bound messages."""

    _handlers: dict[str, Handler] = field(default_factory=dict, repr=False)

    def register(self, msg_type: str, handler: Handler) -> None:
        """Bind *handler* to *msg_type*."""
        self._handlers[msg_type] = handler

    def unregister(self, msg_type: str) -> bool:
        """Remove handler for *msg_type*. Returns True if it existed."""
        if msg_type in self._handlers:
            del self._handlers[msg_type]
            return True
        return False

    def handle(self, message: AgentMessage) -> Any:
        """Dispatch *message* to its registered handler.

        Raises ValueError if no handler is registered for the msg_type.
        """
        handler = self._handlers.get(message.msg_type)
        if handler is None:
            raise ValueError(f"No handler registered for msg_type={message.msg_type!r}")
        return handler(message)

    def can_handle(self, msg_type: str) -> bool:
        return msg_type in self._handlers

    def known_types(self) -> list[str]:
        return list(self._handlers.keys())


# Module-level registry
MESSAGE_HANDLER_REGISTRY: dict[str, MessageHandler] = {}
DEFAULT_MESSAGE_HANDLER = MessageHandler()
MESSAGE_HANDLER_REGISTRY["default"] = DEFAULT_MESSAGE_HANDLER
