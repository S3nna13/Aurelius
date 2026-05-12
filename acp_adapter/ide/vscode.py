"""vscode IDE adapter for ACP protocol."""

from __future__ import annotations

from .base import IDEAdapter


class uvscode(IDEAdapter):
    """uvscode ACP adapter (not yet implemented)."""

    async def connect(self, **kwargs):
        raise NotImplementedError("uvscode is not yet implemented")

    async def send_completion(self, text: str, **kwargs):
        raise NotImplementedError("uvscode is not yet implemented")
