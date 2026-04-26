"""Tests for src/deployment/secret_provider.py."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.deployment.secret_provider import (
    SECRET_PROVIDER_REGISTRY,
    SecretBackend,
    SecretProvider,
    SecretValue,
)

# ---------------------------------------------------------------------------
# SecretBackend enum
# ---------------------------------------------------------------------------


class TestSecretBackend:
    def test_env_value(self):
        assert SecretBackend.ENV == "env"

    def test_file_value(self):
        assert SecretBackend.FILE == "file"

    def test_memory_value(self):
        assert SecretBackend.MEMORY == "memory"

    def test_is_str_subclass(self):
        assert isinstance(SecretBackend.ENV, str)

    def test_members_count(self):
        assert len(SecretBackend) == 3


# ---------------------------------------------------------------------------
# SecretValue dataclass
# ---------------------------------------------------------------------------


class TestSecretValue:
    def test_str_redacted(self):
        sv = SecretValue(key="k", value="secret", redacted=True)
        assert str(sv) == "[REDACTED]"

    def test_str_not_redacted(self):
        sv = SecretValue(key="k", value="my_val", redacted=False)
        assert str(sv) == "my_val"

    def test_default_redacted_false(self):
        sv = SecretValue(key="k", value="v")
        assert sv.redacted is False

    def test_key_stored(self):
        sv = SecretValue(key="mykey", value="v")
        assert sv.key == "mykey"

    def test_value_stored(self):
        sv = SecretValue(key="k", value="myvalue")
        assert sv.value == "myvalue"


# ---------------------------------------------------------------------------
# MEMORY backend
# ---------------------------------------------------------------------------


class TestMemoryBackend:
    def setup_method(self):
        self.provider = SecretProvider(SecretBackend.MEMORY)

    def test_set_get_round_trip(self):
        self.provider.set("FOO", "bar")
        assert self.provider.get("FOO") == "bar"

    def test_get_unknown_returns_none(self):
        assert self.provider.get("DOES_NOT_EXIST") is None

    def test_delete_existing_returns_true(self):
        self.provider.set("KEY1", "val1")
        assert self.provider.delete("KEY1") is True

    def test_delete_nonexistent_returns_false(self):
        assert self.provider.delete("GHOST_KEY") is False

    def test_delete_removes_key(self):
        self.provider.set("TEMP", "v")
        self.provider.delete("TEMP")
        assert self.provider.get("TEMP") is None

    def test_list_keys_includes_set_key(self):
        self.provider.set("LISTED_KEY", "v")
        assert "LISTED_KEY" in self.provider.list_keys()

    def test_list_keys_excludes_deleted(self):
        self.provider.set("DEL_KEY", "v")
        self.provider.delete("DEL_KEY")
        assert "DEL_KEY" not in self.provider.list_keys()

    def test_set_multiple_keys(self):
        self.provider.set("A", "1")
        self.provider.set("B", "2")
        keys = self.provider.list_keys()
        assert "A" in keys
        assert "B" in keys

    def test_overwrite_value(self):
        self.provider.set("OVR", "first")
        self.provider.set("OVR", "second")
        assert self.provider.get("OVR") == "second"

    def test_default_backend_is_memory(self):
        p = SecretProvider()
        assert p._backend == SecretBackend.MEMORY

    def test_redact_true_stores_redacted(self):
        self.provider.set("SECRET", "s3cr3t", redact=True)
        sv = self.provider._store["SECRET"]
        assert sv.redacted is True

    def test_redact_false_stores_not_redacted(self):
        self.provider.set("PLAIN", "plain_value", redact=False)
        sv = self.provider._store["PLAIN"]
        assert sv.redacted is False

    def test_get_returns_actual_value_even_when_redacted(self):
        self.provider.set("S", "actual", redact=True)
        assert self.provider.get("S") == "actual"


# ---------------------------------------------------------------------------
# ENV backend
# ---------------------------------------------------------------------------


class TestEnvBackend:
    def setup_method(self):
        self.provider = SecretProvider(SecretBackend.ENV)
        self._test_keys: list[str] = []

    def teardown_method(self):
        for k in self._test_keys:
            os.environ.pop(k, None)

    def _key(self, name: str) -> str:
        key = f"_AURELIUS_TEST_{name}"
        self._test_keys.append(key)
        return key

    def test_set_puts_into_environ(self):
        k = self._key("SET_ENV")
        self.provider.set(k, "env_val")
        assert os.environ.get(k) == "env_val"

    def test_get_reads_from_environ(self):
        k = self._key("GET_ENV")
        os.environ[k] = "direct_val"
        assert self.provider.get(k) == "direct_val"

    def test_set_get_round_trip(self):
        k = self._key("RT_ENV")
        self.provider.set(k, "round_trip")
        assert self.provider.get(k) == "round_trip"

    def test_get_unknown_returns_none(self):
        assert self.provider.get("_AURELIUS_SURELY_MISSING_XYZ") is None

    def test_delete_removes_from_environ(self):
        k = self._key("DEL_ENV")
        self.provider.set(k, "gone")
        self.provider.delete(k)
        assert os.environ.get(k) is None

    def test_delete_existing_returns_true(self):
        k = self._key("DEL_RET_ENV")
        self.provider.set(k, "v")
        result = self.provider.delete(k)
        assert result is True

    def test_delete_nonexistent_returns_false(self):
        result = self.provider.delete("_AURELIUS_MISSING_KEY_ZZZ")
        assert result is False

    def test_list_keys_contains_set_key(self):
        k = self._key("LIST_ENV")
        self.provider.set(k, "v")
        assert k in self.provider.list_keys()

    def test_list_keys_excludes_deleted(self):
        k = self._key("LIST_DEL_ENV")
        self.provider.set(k, "v")
        self.provider.delete(k)
        assert k not in self.provider.list_keys()

    def test_overwrite_env_value(self):
        k = self._key("OVR_ENV")
        self.provider.set(k, "first")
        self.provider.set(k, "second")
        assert self.provider.get(k) == "second"


# ---------------------------------------------------------------------------
# FILE backend
# ---------------------------------------------------------------------------


class TestFileBackend:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.provider = SecretProvider(SecretBackend.FILE)
        self.provider.set_file_dir(self._tmpdir)

    def test_set_get_round_trip(self):
        self.provider.set("FILE_KEY", "file_value")
        assert self.provider.get("FILE_KEY") == "file_value"

    def test_set_creates_file(self):
        self.provider.set("MY_FILE_KEY", "content")
        p = Path(self._tmpdir) / "MY_FILE_KEY"
        assert p.exists()
        assert p.read_text() == "content"

    def test_list_keys_from_dir(self):
        self.provider.set("K1", "v1")
        self.provider.set("K2", "v2")
        keys = self.provider.list_keys()
        assert "K1" in keys
        assert "K2" in keys

    def test_get_unknown_returns_none(self):
        assert self.provider.get("NONEXISTENT_FILE_KEY") is None

    def test_set_file_dir(self):
        p = SecretProvider(SecretBackend.FILE)
        p.set_file_dir("/tmp")
        assert p._file_dir == "/tmp"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestSecretProviderRegistry:
    def test_has_memory_key(self):
        assert "memory" in SECRET_PROVIDER_REGISTRY

    def test_has_env_key(self):
        assert "env" in SECRET_PROVIDER_REGISTRY

    def test_memory_entry_is_provider(self):
        assert isinstance(SECRET_PROVIDER_REGISTRY["memory"], SecretProvider)

    def test_env_entry_is_provider(self):
        assert isinstance(SECRET_PROVIDER_REGISTRY["env"], SecretProvider)

    def test_memory_backend_is_memory(self):
        assert SECRET_PROVIDER_REGISTRY["memory"]._backend == SecretBackend.MEMORY

    def test_env_backend_is_env(self):
        assert SECRET_PROVIDER_REGISTRY["env"]._backend == SecretBackend.ENV
