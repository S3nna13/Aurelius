"""Simple API key manager with rotation and strong key hashing."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


def _hash_key(raw_key: str, salt: bytes) -> str:
    """Derive a slow hash for *raw_key* using per-key salt."""
    return hashlib.scrypt(
        raw_key.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
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
    _master_key: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)

    def generate(self, name: str, prefix: str = "ak_") -> tuple[str, APIKey]:
        raw = prefix + secrets.token_hex(24)
        key_hash = hmac.new(self._master_key, raw.encode("utf-8"), hashlib.sha256).hexdigest()
        key_prefix = raw[:8]
        key = APIKey(key_hash=key_hash, salt="", name=name, prefix=key_prefix)
        self._keys[key_hash] = key
        return raw, key

    def _find_key(self, raw_key: str) -> APIKey | None:
        key_hash = hmac.new(self._master_key, raw_key.encode("utf-8"), hashlib.sha256).hexdigest()
        return self._keys.get(key_hash)

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
