"""Capability negotiation for the Aurelius protocol surface.

Defines the 6 first-class :class:`Capability` values that Aurelius endpoints
may advertise, plus a :class:`CapabilityNegotiator` that computes the
intersection of client and server capability sets and checks compatibility
against a required set.

Pure stdlib only.  No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Set


class Capability(str, Enum):
    STREAMING = "streaming"
    TOOL_USE = "tool_use"
    VISION = "vision"
    AUDIO = "audio"
    CODE_EXEC = "code_exec"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True)
class CapabilitySet:
    capabilities: FrozenSet[Capability]
    version: str = "1.0"

    def __post_init__(self) -> None:
        # Ensure capabilities is always a frozenset
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


class CapabilityNegotiator:
    """Computes capability agreements between client and server."""

    def negotiate(
        self,
        client_caps: CapabilitySet,
        server_caps: CapabilitySet,
    ) -> CapabilitySet:
        """Return the intersection of *client_caps* and *server_caps*.

        The resulting version is taken from the server capability set.
        """
        return CapabilitySet(
            capabilities=frozenset(
                client_caps.capabilities & server_caps.capabilities
            ),
            version=server_caps.version,
        )

    def is_compatible(
        self,
        client: CapabilitySet,
        server: CapabilitySet,
        required: Set[Capability],
    ) -> bool:
        """Return ``True`` iff every capability in *required* is in the
        negotiated set."""
        negotiated = self.negotiate(client, server)
        return required.issubset(negotiated.capabilities)

    def to_dict(self, cap_set: CapabilitySet) -> Dict[str, Any]:
        """Serialise *cap_set* to a plain dict."""
        return {
            "capabilities": [c.value for c in sorted(cap_set.capabilities, key=lambda c: c.value)],
            "version": cap_set.version,
        }

    def from_dict(self, d: Dict[str, Any]) -> CapabilitySet:
        """Deserialise a :class:`CapabilitySet` from a plain dict."""
        caps = frozenset(Capability(c) for c in d.get("capabilities", []))
        return CapabilitySet(
            capabilities=caps,
            version=d.get("version", "1.0"),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

PROTOCOL_REGISTRY: dict = {
    "capability_negotiator": CapabilityNegotiator(),
}
