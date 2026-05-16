"""Import-time helpers for legacy namespace aliases.

Aurelius historically exposed the same packages through multiple import
prefixes, for example ``src.model``, ``model``, and ``aurelius.model``.  A plain
``sys.modules[prefix] = module`` alias is not enough: later deep imports such as
``aurelius.model.config`` can still load a second module object from the same
file.  This helper installs a small import hook that redirects deep alias imports
back to the canonical ``src.*`` module name.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType

_ALIAS_PREFIXES: dict[str, str] = {}
_FINDER_MARKER = "_aurelius_namespace_alias_finder"


class _NamespaceAliasLoader(importlib.abc.Loader):
    """Loader that returns the canonical module for an alias name."""

    def __init__(self, alias_name: str, canonical_name: str) -> None:
        self.alias_name = alias_name
        self.canonical_name = canonical_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.canonical_name)
        sys.modules[self.alias_name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        return None


class _NamespaceAliasFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that maps legacy deep imports to canonical modules."""

    _aurelius_namespace_alias_finder = True

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        canonical_name = _canonical_for_alias(fullname)
        if canonical_name is None or fullname in sys.modules:
            return None
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None
        is_package = canonical_spec.submodule_search_locations is not None
        loader = _NamespaceAliasLoader(fullname, canonical_name)
        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=canonical_spec.origin,
            is_package=is_package,
        )
        if is_package:
            spec.submodule_search_locations = canonical_spec.submodule_search_locations
        return spec


def _canonical_for_alias(fullname: str) -> str | None:
    for alias_prefix, canonical_prefix in sorted(
        _ALIAS_PREFIXES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if fullname == alias_prefix:
            return canonical_prefix
        if fullname.startswith(f"{alias_prefix}."):
            suffix = fullname[len(alias_prefix) :]
            return f"{canonical_prefix}{suffix}"
    return None


def _ensure_finder_installed() -> None:
    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _NamespaceAliasFinder())


def _mirror_loaded_submodules(canonical_prefix: str, alias_prefix: str) -> None:
    for module_name, module in list(sys.modules.items()):
        if module_name == canonical_prefix or module_name.startswith(f"{canonical_prefix}."):
            alias_name = alias_prefix + module_name[len(canonical_prefix) :]
            sys.modules[alias_name] = module


def register_namespace_aliases(canonical_prefix: str, aliases: tuple[str, ...]) -> None:
    """Register aliases for a canonical package prefix.

    Existing loaded submodules are mirrored immediately, and future deep imports
    through an alias prefix are redirected by the meta-path finder.
    """
    canonical_module = sys.modules.get(canonical_prefix)
    for alias_prefix in aliases:
        _ALIAS_PREFIXES[alias_prefix] = canonical_prefix
        if canonical_module is not None:
            sys.modules[alias_prefix] = canonical_module
        _mirror_loaded_submodules(canonical_prefix, alias_prefix)
    _ensure_finder_installed()
