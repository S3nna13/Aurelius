"""Chat template surface for Aurelius.

Exposes registries so other surfaces (serving, inference, eval) can
look up a named template without importing its module directly.
"""

from __future__ import annotations

from .chatml_template import ChatMLFormatError, ChatMLTemplate, Message

CHAT_TEMPLATE_REGISTRY: dict = {}
MESSAGE_FORMAT_REGISTRY: dict = {}

_chatml = ChatMLTemplate()
CHAT_TEMPLATE_REGISTRY["chatml"] = _chatml
MESSAGE_FORMAT_REGISTRY["chatml"] = Message

__all__ = [
    "CHAT_TEMPLATE_REGISTRY",
    "MESSAGE_FORMAT_REGISTRY",
    "ChatMLFormatError",
    "ChatMLTemplate",
    "Message",
]
