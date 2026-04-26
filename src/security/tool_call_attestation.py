"""Tool-call attestation via HMAC to prevent origin spoofing.

Every tool call emitted by the agent loop is signed with a secret key.
The tool executor verifies the signature before execution.
Fail closed: missing or invalid signature → refusal.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttestedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    nonce: str
    signature: str


class ToolCallAttestation:
    """HMAC-SHA256 attestation for tool calls.

    The secret key is generated per instance. In production, inject a
    persistent key derived from a secrets manager.
    """

    def __init__(self, secret_key: bytes | None = None) -> None:
        self._key = secret_key or secrets.token_bytes(32)

    def _sign(self, tool_name: str, arguments: dict[str, Any], nonce: str) -> str:
        payload = f"{tool_name}:{repr(arguments)}:{nonce}"
        sig = hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    def attest(self, tool_name: str, arguments: dict[str, Any]) -> AttestedToolCall:
        """Sign a tool call and return an attested envelope."""
        nonce = secrets.token_hex(16)
        signature = self._sign(tool_name, arguments, nonce)
        return AttestedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            nonce=nonce,
            signature=signature,
        )

    def verify(self, call: AttestedToolCall) -> bool:
        """Verify the signature on an attested tool call."""
        expected = self._sign(call.tool_name, call.arguments, call.nonce)
        return hmac.compare_digest(expected, call.signature)

    def rotate_key(self) -> bytes:
        """Generate a new key and return the old one for archival."""
        old_key = self._key
        self._key = secrets.token_bytes(32)
        return old_key


# Module-level registry
ATTESTATION_REGISTRY: dict[str, ToolCallAttestation] = {}
DEFAULT_TOOL_CALL_ATTESTATION = ToolCallAttestation()
ATTESTATION_REGISTRY["default"] = DEFAULT_TOOL_CALL_ATTESTATION
