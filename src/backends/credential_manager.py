from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import StrEnum


class ProviderAuthError(Exception):
    pass


class CredentialStatus(StrEnum):
    VALID = "valid"
    EXPIRED = "expired"
    REFRESHING = "refreshing"
    FAILED = "failed"


@dataclass
class Credential:
    provider: str
    token: str
    expires_at: float
    refresh_token: str = ""
    status: CredentialStatus = CredentialStatus.VALID


class CredentialManager:
    """Manages per-provider credentials with auto-refresh on expiry."""

    def __init__(self) -> None:
        self._store: dict[str, Credential] = {}
        self._lock = threading.Lock()

    def register(self, cred: Credential) -> None:
        with self._lock:
            self._store[cred.provider] = cred

    def get(self, provider: str) -> Credential:
        with self._lock:
            if provider not in self._store:
                raise ProviderAuthError(f"no credential registered for provider: {provider}")
            return self._store[provider]

    def is_expired(self, provider: str) -> bool:
        cred = self.get(provider)
        return cred.expires_at > 0 and time.time() > cred.expires_at

    def refresh(self, provider: str) -> Credential:
        # Do not mutate status before raising — caller would be left in REFRESHING forever.
        raise NotImplementedError(
            f"OAuth refresh not implemented for provider: {provider}"
        )

    def get_or_refresh(self, provider: str) -> Credential:
        if self.is_expired(provider):
            try:
                return self.refresh(provider)
            except NotImplementedError:
                raise ProviderAuthError(
                    f"credential for provider '{provider}' is expired "
                    "and refresh is not implemented"
                )
        return self.get(provider)

    def list_providers(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def revoke(self, provider: str) -> None:
        with self._lock:
            if provider not in self._store:
                raise ProviderAuthError(f"no credential registered for provider: {provider}")
            cred = self._store[provider]
            token_bytes = bytearray(cred.token.encode("utf-8"))
            for i in range(len(token_bytes)):
                token_bytes[i] = 0
            cred.token = ""
            cred.refresh_token = ""
            del self._store[provider]


CREDENTIAL_MANAGER = CredentialManager()
