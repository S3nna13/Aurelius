"""Simple API key manager with rotation and hashing."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


@dataclass
class APIKey:
    key_hash: str
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
        raw = prefix + os.urandom(24).hex()
        key_hash = hashlib.sha256(raw.encode()).hexdigest()
        key = APIKey(key_hash=key_hash, name=name, prefix=prefix)
        self._keys[key_hash] = key
        return raw, key

    def validate(self, raw_key: str) -> APIKey | None:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key = self._keys.get(key_hash)
        if key is None or not key.active:
            return None
        return key

    def revoke(self, raw_key: str) -> bool:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key = self._keys.get(key_hash)
        if key is None:
            return False
        key.active = False
        return True

    def list_keys(self) -> list[APIKey]:
        return list(self._keys.values())


API_KEY_MANAGER = APIKeyManager()
