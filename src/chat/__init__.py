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

from .llama3_template import Llama3Template  # noqa: E402
CHAT_TEMPLATE_REGISTRY["llama3"] = Llama3Template()

from .tool_message_formatter import ToolMessageFormatter, ToolResult  # noqa: E402
MESSAGE_FORMAT_REGISTRY["tool_result"] = ToolMessageFormatter

from .harmony_template import (  # noqa: E402
    HarmonyFormatError,
    HarmonyMessage,
    HarmonyTemplate,
)
CHAT_TEMPLATE_REGISTRY["harmony"] = HarmonyTemplate()
MESSAGE_FORMAT_REGISTRY["harmony"] = HarmonyMessage

from .conversation_memory import (  # noqa: E402
    ConversationMemory,
    Fact,
    InMemoryStore,
    JSONFileStore,
)

__all__ += [
    "ConversationMemory",
    "Fact",
    "InMemoryStore",
    "JSONFileStore",
]

from .instruction_tuning_data import (  # noqa: E402
    EvolInstructGenerator,
    InstructionSample,
    MagpieGenerator,
    SelfInstructGenerator,
)

__all__ += [
    "EvolInstructGenerator",
    "InstructionSample",
    "MagpieGenerator",
    "SelfInstructGenerator",
]
