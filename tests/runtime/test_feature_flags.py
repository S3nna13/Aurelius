"""Tests: feature_flags.py — both surface-flag (YAML) and runtime-flag (enum) APIs."""

from __future__ import annotations

import hashlib
import textwrap
import threading

from src.runtime.feature_flags import (
    FEATURE_FLAG_REGISTRY,
    FeatureFlag,
    FeatureFlagRegistry,
    RuntimeFlag,
)

# ===========================================================================
# Part I — Surface-flag (YAML-backed) API  (existing tests preserved)
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "flags.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# ---------------------------------------------------------------------------
# 1. Fail-closed: unknown flag returns False
# ---------------------------------------------------------------------------


def test_unknown_flag_returns_false() -> None:
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("nonexistent_flag") is False


# ---------------------------------------------------------------------------
# 2. register + is_enabled basic
# ---------------------------------------------------------------------------


def test_register_enabled_flag() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="my_feature", enabled=True))
    assert reg.is_enabled("my_feature") is True


def test_register_disabled_flag() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="off_feature", enabled=False))
    assert reg.is_enabled("off_feature") is False


# ---------------------------------------------------------------------------
# 3. list_flags
# ---------------------------------------------------------------------------


def test_list_flags_empty() -> None:
    reg = FeatureFlagRegistry()
    assert reg.list_flags() == []


def test_list_flags_after_register() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="a", enabled=True))
    reg.register(FeatureFlag(name="b", enabled=False))
    names = {f.name for f in reg.list_flags()}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. YAML load
# ---------------------------------------------------------------------------


def test_yaml_load_enabled(tmp_path) -> None:
    path = _make_yaml(
        tmp_path,
        """\
        fast_inference:
          enabled: true
        slow_mode:
          enabled: false
    """,
    )
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("fast_inference") is True
    assert reg.is_enabled("slow_mode") is False


def test_yaml_load_metadata(tmp_path) -> None:
    path = _make_yaml(
        tmp_path,
        """\
        experimental:
          enabled: true
          rollout_pct: 100
          owner: ml-team
    """,
    )
    reg = FeatureFlagRegistry(config_path=path)
    flag = next(f for f in reg.list_flags() if f.name == "experimental")
    assert flag.metadata.get("owner") == "ml-team"


# ---------------------------------------------------------------------------
# 5. env override (surface-flag path)
# ---------------------------------------------------------------------------


def test_env_override_enables_disabled_flag(monkeypatch, tmp_path) -> None:
    path = _make_yaml(
        tmp_path,
        """\
        gated:
          enabled: false
    """,
    )
    monkeypatch.setenv("AURELIUS_FF_GATED", "1")
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("gated") is True


def test_env_override_disables_enabled_flag(monkeypatch, tmp_path) -> None:
    path = _make_yaml(
        tmp_path,
        """\
        always_on:
          enabled: true
    """,
    )
    monkeypatch.setenv("AURELIUS_FF_ALWAYS_ON", "0")
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("always_on") is False


def test_env_override_unknown_flag_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AURELIUS_FF_BRAND_NEW", "1")
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("brand_new") is True


def test_env_override_case_insensitive_name(monkeypatch) -> None:
    monkeypatch.setenv("AURELIUS_FF_MIXED_CASE", "1")
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("mixed_case") is True


# ---------------------------------------------------------------------------
# 6. rollout_pct partial rollout
# ---------------------------------------------------------------------------


def test_rollout_pct_zero_blocks_all() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="zero_rollout", enabled=True, rollout_pct=0.0))
    for uid in ["alice", "bob", "charlie", "dave", "eve"]:
        assert reg.is_enabled("zero_rollout", user_id=uid) is False


def test_rollout_pct_100_allows_all() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="full_rollout", enabled=True, rollout_pct=100.0))
    for uid in ["alice", "bob", "charlie"]:
        assert reg.is_enabled("full_rollout", user_id=uid) is True


def test_rollout_pct_partial_deterministic() -> None:
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="half_rollout", enabled=True, rollout_pct=50.0))
    uid = "deterministic_user"
    bucket = int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 100
    expected = bucket < 50.0
    assert reg.is_enabled("half_rollout", user_id=uid) is expected


# ---------------------------------------------------------------------------
# 7. reload
# ---------------------------------------------------------------------------


def test_reload_picks_up_changes(tmp_path) -> None:
    path = _make_yaml(
        tmp_path,
        """\
        reloadable:
          enabled: false
    """,
    )
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("reloadable") is False

    (tmp_path / "flags.yaml").write_text("reloadable:\n  enabled: true\n")
    reg.reload()
    assert reg.is_enabled("reloadable") is True


# ---------------------------------------------------------------------------
# 8. module-level registry is a FeatureFlagRegistry instance
# ---------------------------------------------------------------------------


def test_module_level_registry_type() -> None:
    assert isinstance(FEATURE_FLAG_REGISTRY, FeatureFlagRegistry)


def test_module_level_registry_fail_closed() -> None:
    assert FEATURE_FLAG_REGISTRY.is_enabled("definitely_not_registered_xyz") is False


# ===========================================================================
# Part II — Runtime-flag (enum-based) API  (new)
# ===========================================================================

# ---------------------------------------------------------------------------
# 9. RuntimeFlag enum members
# ---------------------------------------------------------------------------


class TestRuntimeFlagEnum:
    """Verify the enum has all expected members."""

    def test_all_members_present(self) -> None:
        expected = {
            "MOCK_BACKENDS",
            "EXPERIMENTAL_AGENTS",
            "VERBOSE_LOGGING",
            "TOOL_SANDBOX",
            "ADVANCED_CACHE",
            "TELEMETRY",
        }
        actual = {m.name for m in RuntimeFlag}
        assert actual == expected

    def test_member_values(self) -> None:
        """Each member's value matches its name (used for env-var key construction)."""
        for member in RuntimeFlag:
            assert member.value == member.name


# ---------------------------------------------------------------------------
# 10. Defaults — all False
# ---------------------------------------------------------------------------


class TestRuntimeFlagDefaults:
    """All runtime flags default to False (conservative)."""

    def test_all_default_false(self) -> None:
        reg = FeatureFlagRegistry()
        for flag in RuntimeFlag:
            assert reg.is_enabled(flag) is False, f"{flag.name} should default False"

    def test_all_default_false_via_list_all(self) -> None:
        reg = FeatureFlagRegistry()
        snapshot = reg.list_all()
        for flag in RuntimeFlag:
            assert snapshot[flag.name] is False

    def test_all_default_false_via_to_dict(self) -> None:
        reg = FeatureFlagRegistry()
        d = reg.to_dict()
        for flag in RuntimeFlag:
            assert d[flag.name] is False


# ---------------------------------------------------------------------------
# 11. Env var overrides
# ---------------------------------------------------------------------------


class TestRuntimeFlagEnvOverride:
    def test_env_var_enables_flag(self, monkeypatch) -> None:
        monkeypatch.setenv("AURELIUS_FF_TOOL_SANDBOX", "1")
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.TOOL_SANDBOX) is True

    def test_env_var_disables_flag(self, monkeypatch) -> None:
        monkeypatch.setenv("AURELIUS_FF_MOCK_BACKENDS", "0")
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.MOCK_BACKENDS) is False

    def test_env_var_truthy_values(self, monkeypatch) -> None:
        for truthy in ("1", "true", "True", "TRUE", "yes"):
            monkeypatch.setenv("AURELIUS_FF_TELEMETRY", truthy)
            reg = FeatureFlagRegistry()
            assert reg.is_enabled(RuntimeFlag.TELEMETRY) is True

    def test_env_var_falsy_values(self, monkeypatch) -> None:
        for falsy in ("0", "false", "False", "FALSE", ""):
            monkeypatch.setenv("AURELIUS_FF_VERBOSE_LOGGING", falsy)
            reg = FeatureFlagRegistry()
            assert reg.is_enabled(RuntimeFlag.VERBOSE_LOGGING) is False

    def test_env_var_does_not_affect_other_flags(self, monkeypatch) -> None:
        monkeypatch.setenv("AURELIUS_FF_ADVANCED_CACHE", "1")
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.ADVANCED_CACHE) is True
        # Others should still be False
        assert reg.is_enabled(RuntimeFlag.MOCK_BACKENDS) is False
        assert reg.is_enabled(RuntimeFlag.TELEMETRY) is False

    def test_env_var_multiple_flags(self, monkeypatch) -> None:
        monkeypatch.setenv("AURELIUS_FF_MOCK_BACKENDS", "1")
        monkeypatch.setenv("AURELIUS_FF_TOOL_SANDBOX", "1")
        monkeypatch.setenv("AURELIUS_FF_VERBOSE_LOGGING", "0")
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.MOCK_BACKENDS) is True
        assert reg.is_enabled(RuntimeFlag.TOOL_SANDBOX) is True
        assert reg.is_enabled(RuntimeFlag.VERBOSE_LOGGING) is False
        assert reg.is_enabled(RuntimeFlag.EXPERIMENTAL_AGENTS) is False  # unset


# ---------------------------------------------------------------------------
# 12. Runtime toggling (set)
# ---------------------------------------------------------------------------


class TestRuntimeFlagSet:
    def test_set_enables_flag(self) -> None:
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.TELEMETRY) is False
        reg.set(RuntimeFlag.TELEMETRY, True)
        assert reg.is_enabled(RuntimeFlag.TELEMETRY) is True

    def test_set_disables_flag(self) -> None:
        reg = FeatureFlagRegistry()
        reg.set(RuntimeFlag.TELEMETRY, True)
        reg.set(RuntimeFlag.TELEMETRY, False)
        assert reg.is_enabled(RuntimeFlag.TELEMETRY) is False

    def test_set_only_affects_target_flag(self) -> None:
        reg = FeatureFlagRegistry()
        reg.set(RuntimeFlag.MOCK_BACKENDS, True)
        assert reg.is_enabled(RuntimeFlag.MOCK_BACKENDS) is True
        for flag in RuntimeFlag:
            if flag is not RuntimeFlag.MOCK_BACKENDS:
                assert reg.is_enabled(flag) is False

    def test_set_overrides_env_default(self, monkeypatch) -> None:
        monkeypatch.setenv("AURELIUS_FF_TOOL_SANDBOX", "1")
        reg = FeatureFlagRegistry()
        assert reg.is_enabled(RuntimeFlag.TOOL_SANDBOX) is True
        reg.set(RuntimeFlag.TOOL_SANDBOX, False)
        assert reg.is_enabled(RuntimeFlag.TOOL_SANDBOX) is False


# ---------------------------------------------------------------------------
# 13. list_all and to_dict
# ---------------------------------------------------------------------------


class TestRuntimeFlagListAllToDict:
    def test_list_all_returns_all_flags(self) -> None:
        reg = FeatureFlagRegistry()
        snapshot = reg.list_all()
        assert set(snapshot.keys()) == {m.name for m in RuntimeFlag}
        assert len(snapshot) == len(RuntimeFlag)

    def test_list_all_reflects_changes(self) -> None:
        reg = FeatureFlagRegistry()
        reg.set(RuntimeFlag.VERBOSE_LOGGING, True)
        reg.set(RuntimeFlag.ADVANCED_CACHE, True)
        snapshot = reg.list_all()
        assert snapshot["VERBOSE_LOGGING"] is True
        assert snapshot["ADVANCED_CACHE"] is True
        assert snapshot["MOCK_BACKENDS"] is False

    def test_to_dict_returns_same_as_list_all(self) -> None:
        reg = FeatureFlagRegistry()
        reg.set(RuntimeFlag.TELEMETRY, True)
        assert reg.to_dict() == reg.list_all()

    def test_to_dict_serializable(self) -> None:
        """to_dict() should return a plain dict suitable for JSON serialization."""
        reg = FeatureFlagRegistry()
        d = reg.to_dict()
        assert isinstance(d, dict)
        for k, v in d.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)

    def test_list_all_is_independent_copy(self) -> None:
        """Modifying the returned dict should not affect the registry."""
        reg = FeatureFlagRegistry()
        snapshot = reg.list_all()
        snapshot["MOCK_BACKENDS"] = True
        assert reg.is_enabled(RuntimeFlag.MOCK_BACKENDS) is False


# ---------------------------------------------------------------------------
# 14. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_fef_module_level_registry_is_singleton(self) -> None:
        """FEATURE_FLAG_REGISTRY is the same object when imported multiple times."""
        from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY as reg_a
        from src.runtime.feature_flags import (
            FEATURE_FLAG_REGISTRY as reg_b,
        )

        assert reg_a is reg_b
        assert reg_a is FEATURE_FLAG_REGISTRY

    def test_fef_singleton_is_featureflagregistry(self) -> None:
        assert isinstance(FEATURE_FLAG_REGISTRY, FeatureFlagRegistry)

    def test_fef_singleton_has_runtime_flags(self) -> None:
        """The singleton should have all RuntimeFlag members."""
        from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY as ffr

        for flag in RuntimeFlag:
            # Just check it doesn't raise and returns a bool
            assert isinstance(ffr.is_enabled(flag), bool)


# ---------------------------------------------------------------------------
# 15. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_set_read(self) -> None:
        """Multiple threads should be able to set and read without corruption."""
        reg = FeatureFlagRegistry()
        results: list[bool] = []
        errors: list[Exception] = []

        def toggle(flag: RuntimeFlag, target: bool, repeat: int = 100) -> None:
            try:
                for _ in range(repeat):
                    reg.set(flag, target)
                    # interleave a read
                    _ = reg.is_enabled(flag)
                results.append(reg.is_enabled(flag))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=toggle, args=(RuntimeFlag.MOCK_BACKENDS, True, 50)),
            threading.Thread(target=toggle, args=(RuntimeFlag.MOCK_BACKENDS, False, 50)),
            threading.Thread(target=toggle, args=(RuntimeFlag.TOOL_SANDBOX, True, 50)),
            threading.Thread(target=toggle, args=(RuntimeFlag.TOOL_SANDBOX, False, 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_list_all_and_set(self) -> None:
        """list_all() should not raise under concurrent modification."""
        reg = FeatureFlagRegistry()

        def toggle_and_list() -> None:
            for _ in range(50):
                reg.set(RuntimeFlag.ADVANCED_CACHE, True)
                reg.set(RuntimeFlag.ADVANCED_CACHE, False)
                _ = reg.list_all()

        threads = [threading.Thread(target=toggle_and_list) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # After all threads, list_all should still return valid data
        snapshot = reg.list_all()
        assert set(snapshot.keys()) == {m.name for m in RuntimeFlag}

    def test_concurrent_to_dict(self) -> None:
        """to_dict() should be safe under concurrent writes."""
        reg = FeatureFlagRegistry()

        def writer() -> None:
            for _ in range(50):
                for flag in RuntimeFlag:
                    reg.set(flag, True)
                    reg.set(flag, False)

        def reader() -> None:
            for _ in range(50):
                d = reg.to_dict()
                assert isinstance(d, dict)

        threads = [threading.Thread(target=writer) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
