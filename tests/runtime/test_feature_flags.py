from __future__ import annotations

import hashlib
import os
import textwrap

import pytest

from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag, FeatureFlagRegistry


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

def test_unknown_flag_returns_false():
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("nonexistent_flag") is False


# ---------------------------------------------------------------------------
# 2. register + is_enabled basic
# ---------------------------------------------------------------------------

def test_register_enabled_flag():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="my_feature", enabled=True))
    assert reg.is_enabled("my_feature") is True


def test_register_disabled_flag():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="off_feature", enabled=False))
    assert reg.is_enabled("off_feature") is False


# ---------------------------------------------------------------------------
# 3. list_flags
# ---------------------------------------------------------------------------

def test_list_flags_empty():
    reg = FeatureFlagRegistry()
    assert reg.list_flags() == []


def test_list_flags_after_register():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="a", enabled=True))
    reg.register(FeatureFlag(name="b", enabled=False))
    names = {f.name for f in reg.list_flags()}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. YAML load
# ---------------------------------------------------------------------------

def test_yaml_load_enabled(tmp_path):
    path = _make_yaml(tmp_path, """
        fast_inference:
          enabled: true
        slow_mode:
          enabled: false
    """)
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("fast_inference") is True
    assert reg.is_enabled("slow_mode") is False


def test_yaml_load_metadata(tmp_path):
    path = _make_yaml(tmp_path, """
        experimental:
          enabled: true
          rollout_pct: 100
          owner: ml-team
    """)
    reg = FeatureFlagRegistry(config_path=path)
    flag = next(f for f in reg.list_flags() if f.name == "experimental")
    assert flag.metadata.get("owner") == "ml-team"


# ---------------------------------------------------------------------------
# 5. env override
# ---------------------------------------------------------------------------

def test_env_override_enables_disabled_flag(monkeypatch, tmp_path):
    path = _make_yaml(tmp_path, """
        gated:
          enabled: false
    """)
    monkeypatch.setenv("AURELIUS_FF_GATED", "1")
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("gated") is True


def test_env_override_disables_enabled_flag(monkeypatch, tmp_path):
    path = _make_yaml(tmp_path, """
        always_on:
          enabled: true
    """)
    monkeypatch.setenv("AURELIUS_FF_ALWAYS_ON", "0")
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("always_on") is False


def test_env_override_unknown_flag_enabled(monkeypatch):
    monkeypatch.setenv("AURELIUS_FF_BRAND_NEW", "1")
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("brand_new") is True


def test_env_override_case_insensitive_name(monkeypatch):
    monkeypatch.setenv("AURELIUS_FF_MIXED_CASE", "1")
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("mixed_case") is True


# ---------------------------------------------------------------------------
# 6. rollout_pct partial rollout
# ---------------------------------------------------------------------------

def test_rollout_pct_zero_blocks_all():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="zero_rollout", enabled=True, rollout_pct=0.0))
    for uid in ["alice", "bob", "charlie", "dave", "eve"]:
        assert reg.is_enabled("zero_rollout", user_id=uid) is False


def test_rollout_pct_100_allows_all():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="full_rollout", enabled=True, rollout_pct=100.0))
    for uid in ["alice", "bob", "charlie"]:
        assert reg.is_enabled("full_rollout", user_id=uid) is True


def test_rollout_pct_partial_deterministic():
    reg = FeatureFlagRegistry()
    reg.register(FeatureFlag(name="half_rollout", enabled=True, rollout_pct=50.0))
    uid = "deterministic_user"
    bucket = int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 100
    expected = bucket < 50.0
    assert reg.is_enabled("half_rollout", user_id=uid) is expected


# ---------------------------------------------------------------------------
# 7. reload
# ---------------------------------------------------------------------------

def test_reload_picks_up_changes(tmp_path):
    path = _make_yaml(tmp_path, """
        reloadable:
          enabled: false
    """)
    reg = FeatureFlagRegistry(config_path=path)
    assert reg.is_enabled("reloadable") is False

    (tmp_path / "flags.yaml").write_text("reloadable:\n  enabled: true\n")
    reg.reload()
    assert reg.is_enabled("reloadable") is True


# ---------------------------------------------------------------------------
# 8. module-level registry is a FeatureFlagRegistry instance
# ---------------------------------------------------------------------------

def test_module_level_registry_type():
    assert isinstance(FEATURE_FLAG_REGISTRY, FeatureFlagRegistry)


def test_module_level_registry_fail_closed():
    assert FEATURE_FLAG_REGISTRY.is_enabled("definitely_not_registered_xyz") is False
