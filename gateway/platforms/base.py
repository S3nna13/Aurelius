"""Base platform adapter for gateway messaging."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific message adapters."""

    @abstractmethod
    async def send_message(self, chat_id: str, text: str, **kwargs: Any) -> Any:
        """Send a message to the platform."""
        ...

    @abstractmethod
    async def receive_message(self, **kwargs: Any) -> Any:
        """Receive a message from the platform."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the platform listener."""
        ...
