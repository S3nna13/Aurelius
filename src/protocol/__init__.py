"""Aurelius protocol surface: agent messaging and conversation protocols."""

__all__ = [
    "MessageEnvelope",
    "AgentMessage",
    "MessageBus",
    "MESSAGE_BUS",
    "ConversationProtocol",
    "CONVERSATION_PROTOCOL",
    "StreamingProtocol",
    "STREAMING_PROTOCOL",
    "PROTOCOL_REGISTRY",
]
from .conversation_protocol import CONVERSATION_PROTOCOL, ConversationProtocol
from .message_bus import MESSAGE_BUS, AgentMessage, MessageBus, MessageEnvelope
from .streaming_protocol import STREAMING_PROTOCOL, StreamingProtocol

PROTOCOL_REGISTRY: dict[str, object] = {
    "message_bus": MESSAGE_BUS,
    "conversation": CONVERSATION_PROTOCOL,
    "streaming": STREAMING_PROTOCOL,
}

# --- Delivery tracker (cycle-197) --------------------------------------------
from .delivery_tracker import (  # noqa: F401
    DEFAULT_DELIVERY_TRACKER,
    DELIVERY_TRACKER_REGISTRY,
    DeliveryConfig,
    DeliveryRecord,
    DeliveryStatus,
    DeliveryTracker,
)

PROTOCOL_REGISTRY["delivery_tracker"] = DEFAULT_DELIVERY_TRACKER

# --- Message retry policy (cycle-201) ----------------------------------------
from .message_retry_policy import (  # noqa: F401
    DEFAULT_RETRY_POLICY,
    RETRY_POLICY_REGISTRY,
    BackoffStrategy,
    RetryPolicy,
)

PROTOCOL_REGISTRY["retry_policy"] = DEFAULT_RETRY_POLICY
