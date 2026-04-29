"""Simple API key manager with rotation and strong key hashing."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

_PBKDF2_ROUNDS = 390_000


def _hash_key(raw_key: str, salt: bytes) -> str:
    """Derive a slow hash for *raw_key* using per-key salt."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode("utf-8"),
        salt,
        _PBKDF2_ROUNDS,
    ).hex()


@dataclass
class APIKey:
    key_hash: str
    salt: str
    name: str
    prefix: str = ""
    active: bool = True

    def masked(self) -> str:
        if self.prefix:
            return self.prefix + "***"
        return "***"


@dataclass
class APIKeyManager:
    _keys: dict[str, APIKey] = field(default_factory=dict, repr=False)

    def generate(self, name: str, prefix: str = "ak_") -> tuple[str, APIKey]:
        raw = prefix + secrets.token_hex(24)
        salt = secrets.token_bytes(16)
        key_hash = _hash_key(raw, salt)
        key = APIKey(key_hash=key_hash, salt=salt.hex(), name=name, prefix=prefix)
        self._keys[key_hash] = key
        return raw, key

    def _find_key(self, raw_key: str) -> APIKey | None:
        for key in self._keys.values():
            derived = _hash_key(raw_key, bytes.fromhex(key.salt))
            if hmac.compare_digest(derived, key.key_hash):
                return key
        return None

    def validate(self, raw_key: str) -> APIKey | None:
        key = self._find_key(raw_key)
        if key is None or not key.active:
            return None
        return key

    def revoke(self, raw_key: str) -> bool:
        key = self._find_key(raw_key)
        if key is None:
            return False
        key.active = False
        return True

    def list_keys(self) -> list[APIKey]:
        return list(self._keys.values())


API_KEY_MANAGER = APIKeyManager()
