"""Aurelius MCP capability negotiator.

Provides a lightweight class for advertising, negotiating, and querying
MCP capabilities between local and remote parties.  All logic uses only
stdlib — no external dependencies.
"""

from __future__ import annotations

import re

# Capability name pattern: alphanumeric, dot, or underscore; 1-64 characters.
_CAPABILITY_RE = re.compile(r"^[a-zA-Z0-9._]{1,64}$")


class CapabilityNegotiator:
    """Manages a set of local capabilities and negotiates with a remote party."""

    def __init__(self) -> None:
        self._capabilities: set[str] = set()

    def _validate(self, cap: str) -> None:
        if not isinstance(cap, str):
            raise ValueError(f"capability must be a string, got {type(cap).__name__}")
        if not _CAPABILITY_RE.match(cap):
            raise ValueError(
                f"capability must be non-empty, <=64 chars, and "
                f"alphanumeric/dot/underscore, got {cap!r}"
            )

    def offer_capabilities(self, capabilities: list[str]) -> None:
        """Replace the local capability set with *capabilities*."""
        for cap in capabilities:
            self._validate(cap)
        self._capabilities = set(capabilities)

    def negotiate(self, remote_capabilities: list[str]) -> list[str]:
        """Return the intersection of local and *remote_capabilities*."""
        for cap in remote_capabilities:
            self._validate(cap)
        return sorted(self._capabilities & set(remote_capabilities))

    def can_handle(self, capability: str) -> bool:
        """Return whether *capability* is in the local set."""
        self._validate(capability)
        return capability in self._capabilities

    def list_supported(self) -> list[str]:
        """Return a sorted list of all local capabilities."""
        return sorted(self._capabilities)

    def add_capability(self, cap: str) -> None:
        """Add a single capability to the local set."""
        self._validate(cap)
        self._capabilities.add(cap)

    def remove_capability(self, cap: str) -> None:
        """Remove a single capability from the local set."""
        self._validate(cap)
        self._capabilities.discard(cap)


#: Named collection of registered :class:`CapabilityNegotiator` instances.
CAPABILITY_NEGOTIATOR_REGISTRY: dict[str, CapabilityNegotiator] = {}


__all__ = [
    "CAPABILITY_NEGOTIATOR_REGISTRY",
    "CapabilityNegotiator",
]
