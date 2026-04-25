"""Encryption utility for agent secrets and sensitive config values."""
from __future__ import annotations

import base64
from dataclasses import dataclass


@dataclass
class SimpleEncryptor:
    """Simple symmetric encryption for config secrets using Fernet-compatible AES."""

    key: bytes | None = None

    def __post_init__(self) -> None:
        if self.key is None:
            from cryptography.fernet import Fernet
            self.key = Fernet.generate_key()

    def encrypt(self, plaintext: str) -> str:
        from cryptography.fernet import Fernet
        f = Fernet(self.key)
        return f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        from cryptography.fernet import Fernet
        f = Fernet(self.key)
        return f.decrypt(ciphertext.encode()).decode()


SIMPLE_ENCRYPTOR = SimpleEncryptor()