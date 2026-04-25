"""HMAC-based message signing for Aurelius protocol integrity.

Signs arbitrary byte payloads (or dicts serialised to canonical JSON) with an
HMAC-SHA-256 digest.  Key rotation is supported without recreating the signer.

Pure stdlib only.  No external dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedMessage:
    """Immutable container for a signed payload."""
    payload:   bytes
    signature: str
    algorithm: str
    signed_at: float


@dataclass
class SignerConfig:
    """Mutable configuration so the key can be rotated at runtime."""
    algorithm: str   = "sha256"
    key:       bytes = b"default-key"


# ---------------------------------------------------------------------------
# Signer
# ---------------------------------------------------------------------------


class MessageSigner:
    """Signs and verifies byte payloads using HMAC.

    Usage::

        signer = MessageSigner()
        msg    = signer.sign(b"hello")
        assert signer.verify(msg, b"hello")
    """

    def __init__(self, config: SignerConfig | None = None) -> None:
        self.config: SignerConfig = config if config is not None else SignerConfig()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def sign(self, payload: bytes) -> SignedMessage:
        """Return a :class:`SignedMessage` for *payload*."""
        sig = hmac.new(
            self.config.key,
            payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return SignedMessage(
            payload=payload,
            signature=sig,
            algorithm=self.config.algorithm,
            signed_at=time.monotonic(),
        )

    def verify(self, message: SignedMessage, payload: bytes) -> bool:
        """Return ``True`` iff *message.signature* is valid for *payload*.

        Uses :func:`hmac.compare_digest` for constant-time comparison.
        """
        expected = hmac.new(
            self.config.key,
            payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, message.signature)

    def rotate_key(self, new_key: bytes) -> None:
        """Replace the signing key.  Signatures created with the old key will
        no longer verify."""
        self.config.key = new_key

    # ------------------------------------------------------------------
    # Dict helpers
    # ------------------------------------------------------------------

    def sign_dict(self, data: dict) -> SignedMessage:
        """Serialise *data* to canonical JSON then sign the bytes."""
        payload = json.dumps(data, sort_keys=True).encode()
        return self.sign(payload)

    def verify_dict(self, message: SignedMessage, data: dict) -> bool:
        """Return ``True`` iff *message* is a valid signature for *data*."""
        payload = json.dumps(data, sort_keys=True).encode()
        return self.verify(message, payload)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

MESSAGE_SIGNER_REGISTRY: dict = {
    "default": MessageSigner,
}
