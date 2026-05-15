"""Plugin loader for dynamic agent extension loading.

Uses only stdlib for module discovery and loading from configurable plugin
directories.  Integrates with :class:`PluginManifest` and
:class:`PluginHookRegistry`.
"""

from __future__ import annotations

import importlib.util
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

from src.mcp.plugin_host import PluginManifest

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PluginLoadError(Exception):
    """Raised when plugin loading fails (bad path, missing module, traversal)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LoadedPlugin:
    """A successfully-loaded plugin with runtime metadata."""

    manifest: PluginManifest
    module: types.ModuleType | None
    loaded_at: float
    hooks_registered: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Plugin loader
# ---------------------------------------------------------------------------


@dataclass
class PluginLoader:
    """Discovers and loads plugin modules from filesystem directories."""

    plugin_dirs: list[str] | None = None

    def __post_init__(self) -> None:
        dirs = (
            self.plugin_dirs
            if self.plugin_dirs is not None
            else ["plugins/", "~/.aurelius/plugins/"]
        )
        self._plugin_dirs: list[Path] = [Path(d).expanduser().resolve() for d in dirs]
        self._loaded: dict[str, LoadedPlugin] = {}

    def load(self, plugin_id: str, entry_point: str) -> LoadedPlugin:
        """Load a plugin module by *entry_point* and register it.

        *entry_point* is a dot-separated module path such as
        ``"my_plugin.core"``.  The first segment is treated as the plugin
        directory name inside one of :attr:`_plugin_dirs`.
        """
        if not entry_point or not isinstance(entry_point, str):
            raise PluginLoadError(f"Invalid entry_point: {entry_point!r}")

        parts = entry_point.split(".")
        if ".." in entry_point:
            raise PluginLoadError(f"Path traversal detected in entry_point: {entry_point!r}")

        # Search for the plugin directory in plugin_dirs
        plugin_root: Path | None = None
        for plugin_dir in self._plugin_dirs:
            candidate = plugin_dir / parts[0]
            if candidate.is_dir():
                plugin_root = candidate
                break

        if plugin_root is None:
            raise PluginLoadError(f"Plugin directory for entry point {entry_point!r} not found")

        # Build the file path from remaining parts
        file_path = plugin_root
        for part in parts[1:]:
            file_path = file_path / part

        # Determine the actual Python file to load
        if file_path.is_dir():
            init_file = file_path / "__init__.py"
            if init_file.is_file():
                target = init_file
            else:
                raise PluginLoadError(f"Module path {file_path} is a directory without __init__.py")
        else:
            py_file = file_path.with_suffix(".py")
            if py_file.is_file():
                target = py_file
            else:
                raise PluginLoadError(f"Module file for entry_point {entry_point!r} not found")

        # Path traversal guard: resolved path must stay within plugin_dirs
        resolved = target.resolve()
        if not any(str(resolved).startswith(str(d)) for d in self._plugin_dirs):
            raise PluginLoadError(f"Module path escapes plugin directories: {resolved}")

        # Load the module
        module_name = f"_aurelius_plugin_{plugin_id}_{entry_point.replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Failed to create module spec for {entry_point!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        loaded_at = time.time()
        loaded = LoadedPlugin(
            manifest=PluginManifest(
                plugin_id=plugin_id,
                name=plugin_id,
                version="0.0.0",
                description="",
                entry_point=entry_point,
            ),
            module=module,
            loaded_at=loaded_at,
            hooks_registered=[],
        )
        self._loaded[plugin_id] = loaded
        return loaded

    def unload(self, plugin_id: str) -> None:
        """Remove *plugin_id* from the loaded set."""
        if plugin_id not in self._loaded:
            raise PluginLoadError(f"plugin not found: {plugin_id!r}")
        del self._loaded[plugin_id]

    def get(self, plugin_id: str) -> LoadedPlugin | None:
        """Return the :class:`LoadedPlugin` for *plugin_id*, or ``None``."""
        return self._loaded.get(plugin_id)

    def list_loaded(self) -> list[str]:
        """Return all currently-loaded plugin IDs."""
        return list(self._loaded.keys())

    def is_loaded(self, plugin_id: str) -> bool:
        """Return whether *plugin_id* is currently loaded."""
        return plugin_id in self._loaded

    def reload(self, plugin_id: str, entry_point: str) -> LoadedPlugin:
        """Unload then load *plugin_id*."""
        self.unload(plugin_id)
        return self.load(plugin_id, entry_point)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

DEFAULT_PLUGIN_LOADER: PluginLoader = PluginLoader()

PLUGIN_LOADER_REGISTRY: dict[str, PluginLoader] = {"default": DEFAULT_PLUGIN_LOADER}


__all__ = [
    "DEFAULT_PLUGIN_LOADER",
    "PLUGIN_LOADER_REGISTRY",
    "LoadedPlugin",
    "PluginLoadError",
    "PluginLoader",
]
