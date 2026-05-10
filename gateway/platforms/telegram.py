"""telegram platform adapter for Aurelius gateway."""
from __future__ import annotations

from .base import PlatformAdapter


class utelegram(PlatformAdapter):
    """telegram message adapter (not yet implemented)."""

    async def send_message(self, chat_id: str, text: str, **kwargs):
        raise NotImplementedError("utelegram is not yet implemented")

    async def receive_message(self, **kwargs):
        raise NotImplementedError("utelegram is not yet implemented")

    async def start(self):
        raise NotImplementedError("utelegram is not yet implemented")
