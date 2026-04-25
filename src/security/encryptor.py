"""Encryption utility for agent secrets and sensitive config values."""
from __future__ import annotations

<<<<<<< HEAD
import base64
from dataclasses import dataclass

=======
from dataclasses import dataclass

_HAS_CRYPTO: bool = False
try:
    from cryptography.fernet import Fernet as _Fernet
    _HAS_CRYPTO = True
except ImportError:
    _Fernet = None  # type: ignore

>>>>>>> b0e5923 (sec(cycle-183): encryptor+token_generator (+security×2, STRIDE:InfoDisclosure/Spoofing))

@dataclass
class SimpleEncryptor:
    """Simple symmetric encryption for config secrets using Fernet-compatible AES."""

    key: bytes | None = None

    def __post_init__(self) -> None:
<<<<<<< HEAD
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


try:
    SIMPLE_ENCRYPTOR = SimpleEncryptor()
except ImportError:
    SIMPLE_ENCRYPTOR = None  # cryptography not installed
=======
        if not _HAS_CRYPTO:
            raise RuntimeError("cryptography package required for SimpleEncryptor")
        if self.key is None:
            self.key = _Fernet.generate_key()

    def encrypt(self, plaintext: str) -> str:
        if not _HAS_CRYPTO:
            raise RuntimeError("cryptography package required for SimpleEncryptor")
        f = _Fernet(self.key)
        return f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if not _HAS_CRYPTO:
            raise RuntimeError("cryptography package required for SimpleEncryptor")
        f = _Fernet(self.key)
        return f.decrypt(ciphertext.encode()).decode()


SIMPLE_ENCRYPTOR: SimpleEncryptor | None = SimpleEncryptor() if _HAS_CRYPTO else None
>>>>>>> b0e5923 (sec(cycle-183): encryptor+token_generator (+security×2, STRIDE:InfoDisclosure/Spoofing))
