"""Encryption utility for agent secrets and sensitive config values."""
from __future__ import annotations

import base64
from dataclasses import dataclass


try:
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[misc,assignment]


@dataclass
class SimpleEncryptor:
    """Simple symmetric encryption for config secrets using Fernet-compatible AES."""

    key: bytes | None = None

    def __post_init__(self) -> None:
        if self.key is None:
            if Fernet is None:
                raise ImportError("cryptography is required for SimpleEncryptor")
            self.key = Fernet.generate_key()

    def encrypt(self, plaintext: str) -> str:
        if Fernet is None:
            raise ImportError("cryptography is required for SimpleEncryptor")
        f = Fernet(self.key)
        return f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if Fernet is None:
            raise ImportError("cryptography is required for SimpleEncryptor")
        f = Fernet(self.key)
        return f.decrypt(ciphertext.encode()).decode()


SIMPLE_ENCRYPTOR: "SimpleEncryptor | None" = None
if Fernet is not None:
    SIMPLE_ENCRYPTOR = SimpleEncryptor()