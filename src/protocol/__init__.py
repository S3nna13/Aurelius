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

# --- Delivery tracker (cycle-197) --------------------------------------------
from .delivery_tracker import (  # noqa: F401
    DeliveryConfig,
    DeliveryRecord,
    DeliveryStatus,
    DeliveryTracker,
    DELIVERY_TRACKER_REGISTRY,
    DEFAULT_DELIVERY_TRACKER,
)
PROTOCOL_REGISTRY["delivery_tracker"] = DEFAULT_DELIVERY_TRACKER

# --- Message retry policy (cycle-201) ----------------------------------------
from .message_retry_policy import (  # noqa: F401
    BackoffStrategy,
    RetryPolicy,
    RETRY_POLICY_REGISTRY,
    DEFAULT_RETRY_POLICY,
)
PROTOCOL_REGISTRY["retry_policy"] = DEFAULT_RETRY_POLICY
