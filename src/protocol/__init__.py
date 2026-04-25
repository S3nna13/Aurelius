"""Aurelius protocol surface: agent messaging and conversation protocols."""
__all__ = [
    "MessageEnvelope", "AgentMessage", "MessageBus", "MESSAGE_BUS",
    "ConversationProtocol", "CONVERSATION_PROTOCOL",
    "StreamingProtocol", "STREAMING_PROTOCOL",
    "PROTOCOL_REGISTRY",
]
from .message_bus import MessageEnvelope, AgentMessage, MessageBus, MESSAGE_BUS
from .conversation_protocol import ConversationProtocol, CONVERSATION_PROTOCOL
from .streaming_protocol import StreamingProtocol, STREAMING_PROTOCOL

PROTOCOL_REGISTRY: dict[str, object] = {
    "message_bus": MESSAGE_BUS,
    "conversation": CONVERSATION_PROTOCOL,
    "streaming": STREAMING_PROTOCOL,
}
