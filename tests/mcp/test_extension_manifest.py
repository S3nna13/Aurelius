"""Tests for src.mcp.extension_manifest — ExtensionManifestValidator and friends."""

from __future__ import annotations

from src.mcp.extension_manifest import (
    EXTENSION_MANIFEST_REGISTRY,
    ExtensionManifest,
    ExtensionManifestValidator,
    ManifestValidationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_manifest(**kwargs) -> ExtensionManifest:
    """Return a fully-valid ExtensionManifest, optionally overriding fields."""
    defaults = dict(
        extension_id="my-extension",
        display_name="My Extension",
        version="1.2.3",
        description="A valid extension with enough description length.",
        mcp_version="1.0",
        tools=["tool-a"],
        permissions=["file_read", "network"],
        homepage="https://example.com",
    )
    defaults.update(kwargs)
    return ExtensionManifest(**defaults)


_validator = ExtensionManifestValidator()


# ---------------------------------------------------------------------------
# 1. Valid manifest → ManifestValidationResult(valid=True, errors=[])
# ---------------------------------------------------------------------------


def test_valid_manifest_passes():
    result = _validator.validate(_valid_manifest())
    assert result.valid is True
    assert result.errors == []


def test_valid_manifest_no_warnings_with_known_permissions():
    result = _validator.validate(
        _valid_manifest(permissions=["file_read", "file_write", "network"])
    )
    assert result.valid is True
    assert result.warnings == []


# ---------------------------------------------------------------------------
# 2. Empty extension_id → errors contains message
# ---------------------------------------------------------------------------


def test_empty_extension_id_is_error():
    result = _validator.validate(_valid_manifest(extension_id=""))
    assert result.valid is False
    assert any("extension_id" in e for e in result.errors)


def test_invalid_extension_id_uppercase_is_error():
    result = _validator.validate(_valid_manifest(extension_id="MyExtension"))
    assert result.valid is False
    assert any("extension_id" in e for e in result.errors)


def test_invalid_extension_id_starts_with_digit_is_error():
    result = _validator.validate(_valid_manifest(extension_id="1extension"))
    assert result.valid is False
    assert any("extension_id" in e for e in result.errors)


def test_valid_extension_id_with_hyphens():
    result = _validator.validate(_valid_manifest(extension_id="my-cool-ext"))
    assert result.valid is True
    assert result.errors == []


# ---------------------------------------------------------------------------
# 3. Bad version "1.0" (not semver X.Y.Z) → errors contains message
# ---------------------------------------------------------------------------


def test_bad_version_two_parts_is_error():
    result = _validator.validate(_valid_manifest(version="1.0"))
    assert result.valid is False
    assert any("version" in e for e in result.errors)


def test_bad_version_with_prefix_is_error():
    result = _validator.validate(_valid_manifest(version="v1.0.0"))
    assert result.valid is False
    assert any("version" in e for e in result.errors)


def test_bad_version_with_prerelease_is_error():
    result = _validator.validate(_valid_manifest(version="1.0.0-alpha"))
    assert result.valid is False
    assert any("version" in e for e in result.errors)


def test_good_version_three_parts():
    result = _validator.validate(_valid_manifest(version="2.10.3"))
    assert result.valid is True
    assert result.errors == []


# ---------------------------------------------------------------------------
# 4. Empty display_name → error
# ---------------------------------------------------------------------------


def test_empty_display_name_is_error():
    result = _validator.validate(_valid_manifest(display_name=""))
    assert result.valid is False
    assert any("display_name" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 5. Short description (< 10 chars) → error
# ---------------------------------------------------------------------------


def test_short_description_is_error():
    result = _validator.validate(_valid_manifest(description="Too short"))
    assert result.valid is False
    assert any("description" in e for e in result.errors)


def test_exact_ten_char_description_passes():
    # 10 characters exactly should pass
    result = _validator.validate(_valid_manifest(description="1234567890"))
    assert result.valid is True
    assert not any("description" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 6. Unknown mcp_version "3.0" → valid=True but warnings not empty
# ---------------------------------------------------------------------------


def test_unknown_mcp_version_is_warning_not_error():
    result = _validator.validate(_valid_manifest(mcp_version="3.0"))
    assert result.valid is True
    assert result.errors == []
    assert len(result.warnings) > 0
    assert any("mcp_version" in w or "3.0" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 7. Unknown permission "camera" → error (security fix: reject unknown perms)
# ---------------------------------------------------------------------------


def test_unknown_permission_is_error():
    result = _validator.validate(_valid_manifest(permissions=["camera"]))
    assert result.valid is False
    assert any("camera" in e for e in result.errors)


def test_multiple_unknown_permissions_all_errored():
    result = _validator.validate(_valid_manifest(permissions=["camera", "microphone"]))
    assert result.valid is False
    assert any("camera" in e for e in result.errors)
    assert any("microphone" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 8. All-valid manifest with known permissions → no warnings
# ---------------------------------------------------------------------------


def test_all_known_permissions_no_warnings():
    result = _validator.validate(
        _valid_manifest(
            permissions=["file_read", "file_write", "network", "shell", "env_read"],
            mcp_version="1.0",
        )
    )
    assert result.valid is True
    assert result.warnings == []


# ---------------------------------------------------------------------------
# 9. KNOWN_MCP_VERSIONS contains "1.0"
# ---------------------------------------------------------------------------


def test_known_mcp_versions_contains_1_0():
    assert "1.0" in ExtensionManifestValidator.KNOWN_MCP_VERSIONS


def test_known_mcp_versions_contains_all_expected():
    assert "1.1" in ExtensionManifestValidator.KNOWN_MCP_VERSIONS
    assert "2.0" in ExtensionManifestValidator.KNOWN_MCP_VERSIONS


def test_known_mcp_versions_is_frozenset():
    assert isinstance(ExtensionManifestValidator.KNOWN_MCP_VERSIONS, frozenset)


# ---------------------------------------------------------------------------
# 10. ALLOWED_PERMISSIONS contains "file_read"
# ---------------------------------------------------------------------------


def test_allowed_permissions_contains_file_read():
    assert "file_read" in ExtensionManifestValidator.ALLOWED_PERMISSIONS


def test_allowed_permissions_contains_all_expected():
    expected = {"file_read", "file_write", "network", "shell", "env_read"}
    assert expected <= ExtensionManifestValidator.ALLOWED_PERMISSIONS


def test_allowed_permissions_is_frozenset():
    assert isinstance(ExtensionManifestValidator.ALLOWED_PERMISSIONS, frozenset)


# ---------------------------------------------------------------------------
# 11. ManifestValidationResult dataclass
# ---------------------------------------------------------------------------


def test_manifest_validation_result_fields():
    r = ManifestValidationResult(valid=True, errors=[], warnings=["warn"])
    assert r.valid is True
    assert r.errors == []
    assert r.warnings == ["warn"]


# ---------------------------------------------------------------------------
# 12. EXTENSION_MANIFEST_REGISTRY starts empty (or at least is a dict)
# ---------------------------------------------------------------------------


def test_extension_manifest_registry_is_dict():
    assert isinstance(EXTENSION_MANIFEST_REGISTRY, dict)


# ---------------------------------------------------------------------------
# 13. Multiple errors accumulate
# ---------------------------------------------------------------------------


def test_multiple_errors_accumulate():
    result = _validator.validate(
        _valid_manifest(extension_id="", display_name="", version="bad", description="x")
    )
    assert result.valid is False
    assert len(result.errors) >= 3
