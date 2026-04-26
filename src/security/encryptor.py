"""Encryption utility for agent secrets and sensitive config values."""
from __future__ import annotations

import base64
from dataclasses import dataclass


def _get_fernet():
    """Lazily import cryptography.fernet.Fernet."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for SimpleEncryptor. "
            "Install it with: uv pip install cryptography"
        ) from exc
    return Fernet


@dataclass
class SimpleEncryptor:
    """Simple symmetric encryption for config secrets using Fernet-compatible AES."""

    key: bytes | None = None

    def __post_init__(self) -> None:
        if self.key is None:
            Fernet = _get_fernet()
            self.key = Fernet.generate_key()

    def encrypt(self, plaintext: str) -> str:
        Fernet = _get_fernet()
        f = Fernet(self.key)
        return f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        Fernet = _get_fernet()
        f = Fernet(self.key)
        return f.decrypt(ciphertext.encode()).decode()


# Defer module-level singleton creation to avoid ImportError at import time.
SIMPLE_ENCRYPTOR: SimpleEncryptor | None = None


def _get_default_encryptor() -> SimpleEncryptor:
    global SIMPLE_ENCRYPTOR
    if SIMPLE_ENCRYPTOR is None:
        SIMPLE_ENCRYPTOR = SimpleEncryptor()
    return SIMPLE_ENCRYPTOR
