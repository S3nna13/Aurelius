"""Secret provider abstraction: env-var, file-based, in-memory vault."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SecretBackend(str, Enum):
    ENV = "env"
    FILE = "file"
    MEMORY = "memory"


@dataclass
class SecretValue:
    key: str
    value: str
    redacted: bool = False

    def __str__(self) -> str:
        if self.redacted:
            return "[REDACTED]"
        return self.value


class SecretProvider:
    def __init__(self, backend: SecretBackend = SecretBackend.MEMORY) -> None:
        self._backend = backend
        self._store: dict[str, SecretValue] = {}
        self._file_dir: str | None = None
        # Track keys set via this provider for ENV backend
        self._env_keys: list[str] = []

    def set(self, key: str, value: str, redact: bool = True) -> None:
        sv = SecretValue(key=key, value=value, redacted=redact)
        if self._backend == SecretBackend.ENV:
            os.environ[key] = value
            if key not in self._env_keys:
                self._env_keys.append(key)
        elif self._backend == SecretBackend.FILE:
            self._store[key] = sv
            if self._file_dir is not None:
                file_path = Path(self._file_dir) / key
                file_path.write_text(value)
        else:
            # MEMORY
            self._store[key] = sv

    def get(self, key: str) -> str | None:
        if self._backend == SecretBackend.ENV:
            return os.environ.get(key)
        elif self._backend == SecretBackend.FILE:
            if self._file_dir is not None:
                file_path = Path(self._file_dir) / key
                if file_path.exists():
                    return file_path.read_text()
            # Fall back to in-memory store for FILE backend
            sv = self._store.get(key)
            return sv.value if sv is not None else None
        else:
            # MEMORY
            sv = self._store.get(key)
            return sv.value if sv is not None else None

    def set_file_dir(self, path: str) -> None:
        self._file_dir = path

    def list_keys(self) -> list[str]:
        if self._backend == SecretBackend.ENV:
            return list(self._env_keys)
        elif self._backend == SecretBackend.FILE:
            if self._file_dir is not None:
                try:
                    return [p.name for p in Path(self._file_dir).iterdir() if p.is_file()]
                except FileNotFoundError:
                    return []
            return list(self._store.keys())
        else:
            return list(self._store.keys())

    def delete(self, key: str) -> bool:
        if self._backend == SecretBackend.ENV:
            result = os.environ.pop(key, None)
            existed = result is not None
            if key in self._env_keys:
                self._env_keys.remove(key)
            return existed
        elif self._backend == SecretBackend.FILE:
            existed = key in self._store
            self._store.pop(key, None)
            if self._file_dir is not None:
                file_path = Path(self._file_dir) / key
                if file_path.exists():
                    file_path.unlink()
                    existed = True
            return existed
        else:
            # MEMORY
            if key in self._store:
                del self._store[key]
                return True
            return False


SECRET_PROVIDER_REGISTRY: dict[str, SecretProvider] = {
    "memory": SecretProvider(SecretBackend.MEMORY),
    "env": SecretProvider(SecretBackend.ENV),
}
