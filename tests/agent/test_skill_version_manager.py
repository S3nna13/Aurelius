"""Tests for src.agent.skill_version_manager."""

from __future__ import annotations

import pytest

from src.agent.skill_version_manager import (
    DEFAULT_SKILL_VERSION_MANAGER,
    SKILL_VERSION_REGISTRY,
    SkillVersion,
    SkillVersionError,
    SkillVersionManager,
)


def _fresh() -> SkillVersionManager:
    """Return a fresh manager to avoid test pollution."""
    return SkillVersionManager()


class TestSkillVersionDataclass:
    def test_fields(self) -> None:
        sv = SkillVersion(
            skill_id="s1",
            version="1.0.0",
            checksum="abc123",
            registered_at=1234567890.0,
            metadata={"author": "test"},
        )
        assert sv.skill_id == "s1"
        assert sv.version == "1.0.0"
        assert sv.checksum == "abc123"
        assert sv.registered_at == 1234567890.0
        assert sv.metadata == {"author": "test"}


class TestRegister:
    def test_basic_registration(self) -> None:
        mgr = _fresh()
        result = mgr.register("skill-a", "1.0.0", "chk1")
        assert isinstance(result, SkillVersion)
        assert result.skill_id == "skill-a"
        assert result.version == "1.0.0"
        assert result.checksum == "chk1"
        assert result.metadata == {}
        assert result.registered_at > 0

    def test_registration_with_metadata(self) -> None:
        mgr = _fresh()
        result = mgr.register("skill-b", "2.1", "chk2", metadata={"key": "val"})
        assert result.metadata == {"key": "val"}

    def test_integer_version(self) -> None:
        mgr = _fresh()
        result = mgr.register("skill-c", "42", "chk3")
        assert result.version == "42"

    def test_duplicate_registration_raises(self) -> None:
        mgr = _fresh()
        mgr.register("skill-d", "1.0.0", "chk4")
        with pytest.raises(SkillVersionError, match="already registered"):
            mgr.register("skill-d", "1.0.0", "chk5")


class TestGetLatest:
    def test_returns_latest_semver(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "a")
        mgr.register("skill-a", "1.2.0", "b")
        mgr.register("skill-a", "1.10.0", "c")
        latest = mgr.get_latest("skill-a")
        assert latest is not None
        assert latest.version == "1.10.0"

    def test_returns_latest_integer(self) -> None:
        mgr = _fresh()
        mgr.register("skill-b", "1", "a")
        mgr.register("skill-b", "10", "b")
        mgr.register("skill-b", "2", "c")
        latest = mgr.get_latest("skill-b")
        assert latest is not None
        assert latest.version == "10"

    def test_none_when_skill_missing(self) -> None:
        mgr = _fresh()
        assert mgr.get_latest("missing") is None


class TestGetVersion:
    def test_returns_specific_version(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "a")
        mgr.register("skill-a", "2.0.0", "b")
        result = mgr.get_version("skill-a", "1.0.0")
        assert result is not None
        assert result.checksum == "a"

    def test_none_when_version_missing(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "a")
        assert mgr.get_version("skill-a", "9.9.9") is None


class TestListVersions:
    def test_returns_sorted_versions(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "2.0.0", "a")
        mgr.register("skill-a", "1.0.0", "b")
        mgr.register("skill-a", "1.10.0", "c")
        versions = mgr.list_versions("skill-a")
        assert versions == ["1.0.0", "1.10.0", "2.0.0"]

    def test_empty_list_when_skill_missing(self) -> None:
        mgr = _fresh()
        assert mgr.list_versions("missing") == []


class TestListAll:
    def test_returns_all_skill_ids(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "a")
        mgr.register("skill-b", "1.0.0", "b")
        assert sorted(mgr.list_all()) == ["skill-a", "skill-b"]

    def test_empty_when_no_registrations(self) -> None:
        mgr = _fresh()
        assert mgr.list_all() == []


class TestValidateChecksum:
    def test_valid_checksum(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "abc123")
        assert mgr.validate_checksum("skill-a", "1.0.0", "abc123") is True

    def test_invalid_checksum(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "abc123")
        assert mgr.validate_checksum("skill-a", "1.0.0", "wrong") is False

    def test_missing_skill_returns_false(self) -> None:
        mgr = _fresh()
        assert mgr.validate_checksum("missing", "1.0.0", "abc123") is False

    def test_missing_version_returns_false(self) -> None:
        mgr = _fresh()
        mgr.register("skill-a", "1.0.0", "abc123")
        assert mgr.validate_checksum("skill-a", "2.0.0", "abc123") is False


class TestInvalidInputs:
    def test_empty_skill_id_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="skill_id"):
            mgr.register("", "1.0.0", "chk")

    def test_whitespace_only_skill_id_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="skill_id"):
            mgr.register("   ", "1.0.0", "chk")

    def test_non_string_skill_id_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="skill_id"):
            mgr.register(123, "1.0.0", "chk")  # type: ignore[arg-type]

    def test_empty_version_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="version"):
            mgr.register("skill-a", "", "chk")

    def test_whitespace_only_version_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="version"):
            mgr.register("skill-a", "   ", "chk")

    def test_non_string_version_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="version"):
            mgr.register("skill-a", 1, "chk")  # type: ignore[arg-type]

    def test_invalid_version_format_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="invalid"):
            mgr.register("skill-a", "1.2.3-alpha", "chk")

    def test_version_with_leading_dot_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="invalid"):
            mgr.register("skill-a", ".1.2", "chk")

    def test_version_with_trailing_dot_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="invalid"):
            mgr.register("skill-a", "1.2.", "chk")

    def test_empty_checksum_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="checksum"):
            mgr.register("skill-a", "1.0.0", "")

    def test_whitespace_only_checksum_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="checksum"):
            mgr.register("skill-a", "1.0.0", "   ")

    def test_non_string_checksum_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="checksum"):
            mgr.register("skill-a", "1.0.0", 123)  # type: ignore[arg-type]

    def test_invalid_skill_id_on_get_latest_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="skill_id"):
            mgr.get_latest("")

    def test_invalid_version_on_get_version_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="version"):
            mgr.get_version("skill-a", "bad")

    def test_invalid_version_on_validate_checksum_raises(self) -> None:
        mgr = _fresh()
        with pytest.raises(SkillVersionError, match="version"):
            mgr.validate_checksum("skill-a", "bad", "chk")


class TestSingletons:
    def test_default_singleton_exists(self) -> None:
        assert isinstance(DEFAULT_SKILL_VERSION_MANAGER, SkillVersionManager)

    def test_registry_singleton_exists(self) -> None:
        assert "default" in SKILL_VERSION_REGISTRY
        assert isinstance(SKILL_VERSION_REGISTRY["default"], SkillVersionManager)
