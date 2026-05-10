"""Base IDE adapter for ACP protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IDEAdapter(ABC):
    """Abstract base class for IDE-specific ACP adapters."""

    @abstractmethod
    async def connect(self, **kwargs: Any) -> None:
        """Connect to the IDE."""
        ...

    @abstractmethod
    async def send_completion(self, text: str, **kwargs: Any) -> Any:
        """Send a completion to the IDE."""
        ...
