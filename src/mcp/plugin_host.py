"""Aurelius MCP plugin host and extension registry.

Provides hot-reload capable plugin/extension management for the Aurelius MCP
surface.  All logic uses only stdlib — no external dependencies.

Inspired by Cline MCP integration (MIT, github.com/cline/cline), Goose
extension/MCP system (Apache-2.0, github.com/block/goose), clean-room
reimplementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Semantic version pattern: X.Y.Z (integers only, no pre-release suffix)
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PluginManifest:
    """Descriptor for a single Aurelius plugin/extension."""

    plugin_id: str
    name: str
    version: str
    description: str
    tool_schemas: list[str] = field(default_factory=list)
    entry_point: str = ""
    enabled: bool = True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PluginHostError(Exception):
    """Raised for invalid plugin operations (bad manifest, unknown id, etc.)."""


# ---------------------------------------------------------------------------
# Plugin host
# ---------------------------------------------------------------------------


class PluginHost:
    """Registry of :class:`PluginManifest` objects with lifecycle management.

    Supports register, unregister, enable/disable, hot-reload, and serialization.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginManifest] = {}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_manifest(manifest: PluginManifest) -> None:
        if not isinstance(manifest.plugin_id, str) or not manifest.plugin_id.strip():
            raise PluginHostError(
                "plugin_id must be a non-empty string, "
                f"got {manifest.plugin_id!r}"
            )
        if not _VERSION_RE.match(manifest.version):
            raise PluginHostError(
                f"version must match 'X.Y.Z' (integers), got {manifest.version!r}"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, manifest: PluginManifest) -> None:
        """Validate and register *manifest*.

        Raises :class:`PluginHostError` if validation fails.
        """
        self._validate_manifest(manifest)
        self._plugins[manifest.plugin_id] = manifest

    def unregister(self, plugin_id: str) -> None:
        """Remove the plugin with *plugin_id*.

        Raises :class:`PluginHostError` if not found.
        """
        if plugin_id not in self._plugins:
            raise PluginHostError(f"plugin not found: {plugin_id!r}")
        del self._plugins[plugin_id]

    def get(self, plugin_id: str) -> PluginManifest:
        """Return the manifest for *plugin_id*.

        Raises :class:`PluginHostError` if not found.
        """
        if plugin_id not in self._plugins:
            raise PluginHostError(f"plugin not found: {plugin_id!r}")
        return self._plugins[plugin_id]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_plugins(self, enabled_only: bool = True) -> list[PluginManifest]:
        """Return all registered plugins, optionally filtered to enabled ones."""
        plugins = list(self._plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.enabled]
        return plugins

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self, plugin_id: str) -> None:
        """Mark the plugin as enabled.

        Raises :class:`PluginHostError` if not found.
        """
        self.get(plugin_id).enabled = True

    def disable(self, plugin_id: str) -> None:
        """Mark the plugin as disabled.

        Raises :class:`PluginHostError` if not found.
        """
        self.get(plugin_id).enabled = False

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self, plugin_id: str, new_manifest: PluginManifest) -> None:
        """Replace the manifest for *plugin_id* in place (hot-reload simulation).

        Validates *new_manifest* before replacing.  Raises :class:`PluginHostError`
        if *plugin_id* is not registered or *new_manifest* is invalid.
        """
        if plugin_id not in self._plugins:
            raise PluginHostError(f"plugin not found: {plugin_id!r}")
        self._validate_manifest(new_manifest)
        self._plugins[plugin_id] = new_manifest

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of all registered plugins."""
        return {
            pid: {
                "plugin_id": m.plugin_id,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "tool_schemas": list(m.tool_schemas),
                "entry_point": m.entry_point,
                "enabled": m.enabled,
            }
            for pid, m in self._plugins.items()
        }


# ---------------------------------------------------------------------------
# Registries and singleton
# ---------------------------------------------------------------------------

#: Named collection of :class:`PluginHost` instances.
PLUGIN_HOST_REGISTRY: dict[str, PluginHost] = {}

#: Default singleton host used when no specific host is needed.
DEFAULT_PLUGIN_HOST: PluginHost = PluginHost()

# Seed the named registry with the default host.
PLUGIN_HOST_REGISTRY["default"] = DEFAULT_PLUGIN_HOST


__all__ = [
    "DEFAULT_PLUGIN_HOST",
    "PLUGIN_HOST_REGISTRY",
    "PluginHost",
    "PluginHostError",
    "PluginManifest",
]
