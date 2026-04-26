"""Tests for src.agent.plugin_loader — dynamic plugin module loading."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.agent.plugin_loader import (
    DEFAULT_PLUGIN_LOADER,
    PLUGIN_LOADER_REGISTRY,
    LoadedPlugin,
    PluginLoadError,
    PluginLoader,
)
from src.mcp.plugin_host import PluginManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_dir(tmp_path: Path, plugin_name: str = "my_plugin") -> Path:
    """Create a temporary plugin directory with __init__.py and core.py."""
    plugin_dir = tmp_path / plugin_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        '"""My plugin package."""\nPLUGIN_NAME = "my_plugin"\n',
        encoding="utf-8",
    )
    core_file = plugin_dir / "core.py"
    core_file.write_text(
        '"""My plugin core."""\nMY_VAR = 42\n\ndef hello():\n    return "hello"\n',
        encoding="utf-8",
    )
    return plugin_dir


# ---------------------------------------------------------------------------
# PluginLoadError
# ---------------------------------------------------------------------------


class TestPluginLoadError:
    def test_is_exception(self):
        assert issubclass(PluginLoadError, Exception)


# ---------------------------------------------------------------------------
# LoadedPlugin
# ---------------------------------------------------------------------------


class TestLoadedPlugin:
    def test_dataclass_fields(self):
        manifest = PluginManifest(
            plugin_id="test", name="Test", version="1.0.0", description="desc"
        )
        lp = LoadedPlugin(
            manifest=manifest,
            module=None,
            loaded_at=123.0,
            hooks_registered=["pre_tool_call"],
        )
        assert lp.manifest is manifest
        assert lp.module is None
        assert lp.loaded_at == 123.0
        assert lp.hooks_registered == ["pre_tool_call"]


# ---------------------------------------------------------------------------
# PluginLoader init
# ---------------------------------------------------------------------------


class TestPluginLoaderInit:
    def test_default_plugin_dirs(self):
        loader = PluginLoader()
        assert len(loader._plugin_dirs) == 2
        assert str(loader._plugin_dirs[0]).endswith("plugins")
        assert ".aurelius" in str(loader._plugin_dirs[1])

    def test_custom_plugin_dirs(self, tmp_path: Path):
        custom = str(tmp_path / "custom_plugins")
        loader = PluginLoader(plugin_dirs=[custom])
        assert len(loader._plugin_dirs) == 1
        assert loader._plugin_dirs[0] == (tmp_path / "custom_plugins").resolve()

    def test_expanduser_in_plugin_dirs(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        loader = PluginLoader(plugin_dirs=["~/my_plugins"])
        assert loader._plugin_dirs[0] == (home / "my_plugins").resolve()


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_valid_plugin(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        result = loader.load("my_plugin", "my_plugin.core")

        assert isinstance(result, LoadedPlugin)
        assert result.manifest.plugin_id == "my_plugin"
        assert result.module is not None
        assert hasattr(result.module, "hello")
        assert result.module.hello() == "hello"
        assert "my_plugin" in loader.list_loaded()

    def test_load_package_init(self, tmp_path: Path):
        """Loading entry_point that points to a package __init__.py."""
        plugin_dir = tmp_path / "pkg_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("X = 1\n", encoding="utf-8")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        result = loader.load("pkg", "pkg_plugin")
        assert result.module is not None
        assert result.module.X == 1

    def test_load_missing_directory_raises(self, tmp_path: Path):
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        with pytest.raises(PluginLoadError, match="not found"):
            loader.load("missing", "missing.core")

    def test_load_missing_module_raises(self, tmp_path: Path):
        plugin_dir = tmp_path / "partial"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        with pytest.raises(PluginLoadError, match="not found"):
            loader.load("partial", "partial.nonexistent")

    def test_load_path_traversal_dotdot_in_entry_point(self, tmp_path: Path):
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        with pytest.raises(PluginLoadError, match="traversal"):
            loader.load("bad", "my_plugin..core")

    def test_load_path_traversal_escapes_plugin_dir(self, tmp_path: Path):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text('SECRET = "leaked"\n', encoding="utf-8")
        (plugin_dir / "escape").symlink_to(outside)

        loader = PluginLoader(plugin_dirs=[str(plugin_dir)])
        with pytest.raises(PluginLoadError, match="escapes"):
            loader.load("escape", "escape.secret")

    def test_load_invalid_entry_point_empty(self):
        loader = PluginLoader(plugin_dirs=[])
        with pytest.raises(PluginLoadError, match="Invalid"):
            loader.load("x", "")

    def test_load_invalid_entry_point_none(self):
        loader = PluginLoader(plugin_dirs=[])
        with pytest.raises(PluginLoadError, match="Invalid"):
            loader.load("x", None)  # type: ignore[arg-type]

    def test_load_records_loaded_at(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        before = time.time()
        result = loader.load("my_plugin", "my_plugin.core")
        after = time.time()
        assert before <= result.loaded_at <= after

    def test_load_stores_in_loaded_dict(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        result = loader.load("my_plugin", "my_plugin.core")
        assert loader.get("my_plugin") is result


# ---------------------------------------------------------------------------
# unload / get / list_loaded / is_loaded
# ---------------------------------------------------------------------------


class TestUnloadGetListIsLoaded:
    def test_unload_removes_plugin(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        loader.load("my_plugin", "my_plugin.core")
        loader.unload("my_plugin")
        assert loader.get("my_plugin") is None
        assert not loader.is_loaded("my_plugin")

    def test_unload_unknown_raises(self):
        loader = PluginLoader(plugin_dirs=[])
        with pytest.raises(PluginLoadError, match="not found"):
            loader.unload("ghost")

    def test_get_returns_none_for_unknown(self):
        loader = PluginLoader(plugin_dirs=[])
        assert loader.get("unknown") is None

    def test_list_loaded_returns_ids(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "plugin_a")
        _make_plugin_dir(tmp_path, "plugin_b")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        loader.load("a", "plugin_a.core")
        loader.load("b", "plugin_b.core")
        ids = loader.list_loaded()
        assert sorted(ids) == ["a", "b"]

    def test_list_loaded_empty(self):
        loader = PluginLoader(plugin_dirs=[])
        assert loader.list_loaded() == []

    def test_is_loaded_true(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        loader.load("my_plugin", "my_plugin.core")
        assert loader.is_loaded("my_plugin") is True

    def test_is_loaded_false(self):
        loader = PluginLoader(plugin_dirs=[])
        assert loader.is_loaded("nobody") is False


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_unloads_and_loads(self, tmp_path: Path):
        _make_plugin_dir(tmp_path, "my_plugin")
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        first = loader.load("my_plugin", "my_plugin.core")
        first_time = first.loaded_at

        time.sleep(0.01)
        second = loader.reload("my_plugin", "my_plugin.core")
        assert second.loaded_at > first_time
        assert loader.get("my_plugin") is second

    def test_reload_unknown_raises(self, tmp_path: Path):
        loader = PluginLoader(plugin_dirs=[str(tmp_path)])
        with pytest.raises(PluginLoadError, match="not found"):
            loader.reload("ghost", "ghost.core")


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestSingletons:
    def test_default_plugin_loader_is_plugin_loader(self):
        assert isinstance(DEFAULT_PLUGIN_LOADER, PluginLoader)

    def test_plugin_loader_registry_is_dict(self):
        assert isinstance(PLUGIN_LOADER_REGISTRY, dict)

    def test_plugin_loader_registry_has_default(self):
        assert "default" in PLUGIN_LOADER_REGISTRY

    def test_plugin_loader_registry_default_is_default_loader(self):
        assert PLUGIN_LOADER_REGISTRY["default"] is DEFAULT_PLUGIN_LOADER

    def test_custom_loader_in_registry(self):
        custom = PluginLoader(plugin_dirs=[])
        PLUGIN_LOADER_REGISTRY["custom"] = custom
        assert PLUGIN_LOADER_REGISTRY["custom"] is custom
        del PLUGIN_LOADER_REGISTRY["custom"]


# ---------------------------------------------------------------------------
# Loading outside plugin_dirs
# ---------------------------------------------------------------------------


class TestLoadingOutsidePluginDirs:
    def test_load_outside_plugin_dirs_is_rejected(self, tmp_path: Path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "bad_plugin").mkdir()
        (outside_dir / "bad_plugin" / "core.py").write_text("x = 1\n", encoding="utf-8")

        loader = PluginLoader(plugin_dirs=[str(plugins_dir)])
        with pytest.raises(PluginLoadError, match="not found"):
            loader.load("bad", "bad_plugin.core")
