from __future__ import annotations

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

    def register(self, cred: Credential) -> None:
        self._store[cred.provider] = cred

    def get(self, provider: str) -> Credential:
        if provider not in self._store:
            raise ProviderAuthError(f"no credential registered for provider: {provider}")
        return self._store[provider]

    def is_expired(self, provider: str) -> bool:
        cred = self.get(provider)
        return cred.expires_at > 0 and time.time() > cred.expires_at

    def refresh(self, provider: str) -> Credential:
        cred = self.get(provider)
        cred.status = CredentialStatus.REFRESHING
        new_token = f"refreshed_{provider}_{int(time.time())}"
        cred.token = new_token
        cred.expires_at = time.time() + 3600
        cred.status = CredentialStatus.VALID
        return cred

    def get_or_refresh(self, provider: str) -> Credential:
        if self.is_expired(provider):
            return self.refresh(provider)
        return self.get(provider)

    def list_providers(self) -> list[str]:
        return list(self._store.keys())

    def revoke(self, provider: str) -> None:
        if provider not in self._store:
            raise ProviderAuthError(f"no credential registered for provider: {provider}")
        del self._store[provider]


CREDENTIAL_MANAGER = CredentialManager()
