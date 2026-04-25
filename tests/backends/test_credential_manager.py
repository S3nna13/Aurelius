import time
import pytest
from src.backends.credential_manager import (
    CREDENTIAL_MANAGER,
    Credential,
    CredentialManager,
    CredentialStatus,
    ProviderAuthError,
)


def fresh_cred(provider="test_provider", expires_at=0.0):
    return Credential(provider=provider, token="tok_abc", expires_at=expires_at)


# --- CredentialManager unit tests ---

def test_register_and_get():
    cm = CredentialManager()
    cred = fresh_cred("p1")
    cm.register(cred)
    assert cm.get("p1") is cred


def test_get_missing_raises():
    cm = CredentialManager()
    with pytest.raises(ProviderAuthError):
        cm.get("nonexistent")


def test_is_expired_never_expires():
    cm = CredentialManager()
    cm.register(fresh_cred("p2", expires_at=0.0))
    assert cm.is_expired("p2") is False


def test_is_expired_past_timestamp():
    cm = CredentialManager()
    cm.register(fresh_cred("p3", expires_at=1.0))
    assert cm.is_expired("p3") is True


def test_is_expired_future_timestamp():
    cm = CredentialManager()
    cm.register(fresh_cred("p4", expires_at=time.time() + 3600))
    assert cm.is_expired("p4") is False


def test_refresh_generates_new_token():
    cm = CredentialManager()
    cm.register(fresh_cred("p5"))
    refreshed = cm.refresh("p5")
    assert refreshed.token.startswith("refreshed_p5_")


def test_refresh_sets_valid_status():
    cm = CredentialManager()
    cm.register(fresh_cred("p6"))
    refreshed = cm.refresh("p6")
    assert refreshed.status == CredentialStatus.VALID


def test_refresh_extends_expires_at():
    cm = CredentialManager()
    before = time.time()
    cm.register(fresh_cred("p7", expires_at=1.0))
    cm.refresh("p7")
    after = time.time()
    cred = cm.get("p7")
    assert before + 3600 <= cred.expires_at <= after + 3600 + 1


def test_get_or_refresh_valid_credential():
    cm = CredentialManager()
    cred = fresh_cred("p8", expires_at=time.time() + 3600)
    original_token = cred.token
    cm.register(cred)
    result = cm.get_or_refresh("p8")
    assert result.token == original_token


def test_get_or_refresh_expired_credential():
    cm = CredentialManager()
    cm.register(fresh_cred("p9", expires_at=1.0))
    result = cm.get_or_refresh("p9")
    assert result.token.startswith("refreshed_p9_")


def test_list_providers():
    cm = CredentialManager()
    for name in ("a", "b", "c"):
        cm.register(fresh_cred(name))
    providers = cm.list_providers()
    assert set(providers) == {"a", "b", "c"}


def test_revoke_removes_credential():
    cm = CredentialManager()
    cm.register(fresh_cred("p10"))
    cm.revoke("p10")
    with pytest.raises(ProviderAuthError):
        cm.get("p10")


def test_revoke_missing_raises():
    cm = CredentialManager()
    with pytest.raises(ProviderAuthError):
        cm.revoke("ghost")


def test_module_level_credential_manager_singleton():
    assert isinstance(CREDENTIAL_MANAGER, CredentialManager)


def test_credential_default_status():
    cred = Credential(provider="x", token="t", expires_at=0.0)
    assert cred.status == CredentialStatus.VALID


def test_register_overwrites_existing():
    cm = CredentialManager()
    cm.register(Credential(provider="dup", token="old", expires_at=0.0))
    cm.register(Credential(provider="dup", token="new", expires_at=0.0))
    assert cm.get("dup").token == "new"
