"""zed IDE adapter for ACP protocol."""
from __future__ import annotations

from .base import IDEAdapter


class uzed(IDEAdapter):
    """uzed ACP adapter (not yet implemented)."""

    async def connect(self, **kwargs):
        raise NotImplementedError("uzed is not yet implemented")

    async def send_completion(self, text: str, **kwargs):
        raise NotImplementedError("uzed is not yet implemented")
