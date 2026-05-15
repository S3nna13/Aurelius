"""whatsapp platform adapter for Aurelius gateway."""

from __future__ import annotations

from .base import PlatformAdapter


class uwhatsapp(PlatformAdapter):
    """whatsapp message adapter (not yet implemented)."""

    async def send_message(self, chat_id: str, text: str, **kwargs):
        raise NotImplementedError("uwhatsapp is not yet implemented")

    async def receive_message(self, **kwargs):
        raise NotImplementedError("uwhatsapp is not yet implemented")

    async def start(self):
        raise NotImplementedError("uwhatsapp is not yet implemented")
