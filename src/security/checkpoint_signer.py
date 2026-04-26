"""Detached signature verification for model checkpoints.

Supports Ed25519 signing (via the ``cryptography`` library) with automatic
HMAC-SHA256 fallback.  Every checkpoint is hashed with SHA-256 before
signing so large weights are never loaded into the signing path directly.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint signature is missing, malformed, or invalid."""


@dataclass
class CheckpointSigner:
    """Signs and verifies checkpoint SHA-256 digests with Ed25519 or HMAC."""

    private_key_bytes: bytes = b""
    public_key_bytes: bytes = b""
    hmac_key: bytes = b""
    algorithm: Literal["ed25519", "hmac-sha256"] = "hmac-sha256"

    def __post_init__(self) -> None:
        if self.private_key_bytes or self.public_key_bytes:
            self._use_ed25519()
        else:
            self.algorithm = "hmac-sha256"

    def sign_digest(self, sha256_hex: str) -> str:
        raw = bytes.fromhex(sha256_hex)
        if self.algorithm == "ed25519":
            return self._ed25519_sign(raw)
        return self._hmac_sign(raw)

    def verify_digest(self, sha256_hex: str, signature_hex: str) -> bool:
        raw = bytes.fromhex(sha256_hex)
        sig = bytes.fromhex(signature_hex)
        try:
            if self.algorithm == "ed25519":
                ok = self._ed25519_verify(raw, sig)
            else:
                ok = self._hmac_verify(raw, sig)
        except Exception as exc:
            raise CheckpointIntegrityError(str(exc)) from exc
        if not ok:
            raise CheckpointIntegrityError("signature does not match checkpoint digest")
        return True

    def sign_file(self, file_path: str) -> str:
        sha256_hex = self._hash_file(file_path)
        return self.sign_digest(sha256_hex)

    def verify_file(self, file_path: str, signature_hex: str) -> bool:
        sha256_hex = self._hash_file(file_path)
        return self.verify_digest(sha256_hex, signature_hex)

    def _hmac_sign(self, raw: bytes) -> str:
        return hmac.new(self.hmac_key, raw, hashlib.sha256).hexdigest()

    def _hmac_verify(self, raw: bytes, sig: bytes) -> bool:
        expected = self._hmac_sign(raw)
        return hmac.compare_digest(expected, sig.hex())

    def _use_ed25519(self) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError:
            logger.warning("cryptography library unavailable; falling back to HMAC-SHA256")
            self.algorithm = "hmac-sha256"
            return
        self.algorithm = "ed25519"
        self._ed25519_curve = ed25519

    def _ed25519_sign(self, raw: bytes) -> str:
        if not self.private_key_bytes:
            raise CheckpointIntegrityError("Ed25519 private key not configured")
        private_key = self._ed25519_curve.Ed25519PrivateKey.from_private_bytes(
            self.private_key_bytes
        )
        return private_key.sign(raw).hex()

    def _ed25519_verify(self, raw: bytes, sig: bytes) -> bool:
        if not self.public_key_bytes:
            raise CheckpointIntegrityError("Ed25519 public key not configured")
        public_key = self._ed25519_curve.Ed25519PublicKey.from_public_bytes(self.public_key_bytes)
        try:
            public_key.verify(sig, raw)
            return True
        except Exception:
            return False

    @staticmethod
    def _hash_file(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


CHECKPOINT_SIGNER_REGISTRY: dict[str, CheckpointSigner] = {}
DEFAULT_CHECKPOINT_SIGNER = CheckpointSigner(hmac_key=b"default-dev-key-change-me")
CHECKPOINT_SIGNER_REGISTRY["default"] = DEFAULT_CHECKPOINT_SIGNER
