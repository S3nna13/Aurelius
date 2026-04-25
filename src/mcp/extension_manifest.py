"""Aurelius MCP extension manifest validator.

Provides dataclasses and validation logic for MCP extension manifests,
covering field format, semver, known MCP versions, and permission allow-lists.
All logic uses only stdlib — no external dependencies.

Inspired by Anthropic Claude Code skills system (MIT), Goose extension/MCP
(Apache-2.0), clean-room reimplementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Extension ID pattern: starts with lowercase letter, followed by lowercase
# alphanumeric characters or hyphens.
_EXTENSION_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# Strict semantic version: X.Y.Z (integers only, no pre-release suffix)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExtensionManifestError(Exception):
    """Raised for unrecoverable extension manifest errors."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExtensionManifest:
    """Descriptor for a single MCP extension."""

    extension_id: str
    display_name: str
    version: str
    description: str
    mcp_version: str = "1.0"
    tools: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    homepage: str | None = None


@dataclass
class ManifestValidationResult:
    """Outcome of validating an :class:`ExtensionManifest`."""

    valid: bool
    errors: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ExtensionManifestValidator:
    """Validates :class:`ExtensionManifest` instances against MCP conventions.

    Produces a :class:`ManifestValidationResult` detailing any errors
    (blocking) and warnings (non-blocking).
    """

    #: MCP protocol versions that are explicitly recognised.
    KNOWN_MCP_VERSIONS: frozenset[str] = frozenset({"1.0", "1.1", "2.0"})

    #: Permissions that extensions are permitted to declare.
    ALLOWED_PERMISSIONS: frozenset[str] = frozenset(
        {"file_read", "file_write", "network", "shell", "env_read"}
    )

    def validate(self, manifest: ExtensionManifest) -> ManifestValidationResult:
        """Validate *manifest* and return a :class:`ManifestValidationResult`.

        Errors mean the manifest is invalid; warnings are advisory only.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # --- extension_id ---
        if not manifest.extension_id:
            errors.append("extension_id must be a non-empty string")
        elif not _EXTENSION_ID_RE.match(manifest.extension_id):
            errors.append(
                f"extension_id must match '^[a-z][a-z0-9-]*$', "
                f"got {manifest.extension_id!r}"
            )

        # --- version (strict semver X.Y.Z) ---
        if not _SEMVER_RE.match(manifest.version):
            errors.append(
                f"version must match semantic versioning 'X.Y.Z', "
                f"got {manifest.version!r}"
            )

        # --- display_name ---
        if not manifest.display_name or not manifest.display_name.strip():
            errors.append("display_name must be a non-empty string")

        # --- description ---
        if not manifest.description or len(manifest.description) < 10:
            errors.append(
                "description must be a non-empty string with at least 10 characters"
            )

        # --- mcp_version ---
        if manifest.mcp_version not in self.KNOWN_MCP_VERSIONS:
            warnings.append(
                f"mcp_version {manifest.mcp_version!r} is not in the list of "
                f"known MCP versions {sorted(self.KNOWN_MCP_VERSIONS)!r}; "
                "validation may be incomplete"
            )

        # --- permissions ---
        for perm in manifest.permissions:
            if perm not in self.ALLOWED_PERMISSIONS:
                warnings.append(
                    f"permission {perm!r} is not in the list of "
                    f"allowed permissions {sorted(self.ALLOWED_PERMISSIONS)!r}"
                )

        valid = len(errors) == 0
        return ManifestValidationResult(valid=valid, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Named manifest registry
# ---------------------------------------------------------------------------

#: Named collection of registered :class:`ExtensionManifest` instances.
EXTENSION_MANIFEST_REGISTRY: dict[str, ExtensionManifest] = {}


__all__ = [
    "EXTENSION_MANIFEST_REGISTRY",
    "ExtensionManifest",
    "ExtensionManifestError",
    "ExtensionManifestValidator",
    "ManifestValidationResult",
]
