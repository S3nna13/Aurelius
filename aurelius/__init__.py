"""
Aurelius AI compatibility package.

This package keeps legacy ``aurelius.*`` imports connected to the
``src.*`` package tree and the repo-root compatibility modules without
duplicating the source layout.
"""

from __future__ import annotations

import os
from importlib import import_module

__version__ = "1.0.0"

_PACKAGE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, os.pardir))
_SRC_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, os.pardir, "src"))

_package_path = list(__path__)
if os.path.isdir(_REPO_ROOT):
    _package_path.insert(0, _REPO_ROOT)
if os.path.isdir(_SRC_ROOT):
    _package_path.insert(0, _SRC_ROOT)
__path__ = _package_path

_PUBLIC_SUBPACKAGES = (
    "agent",
    "alignment",
    "backends",
    "chat",
    "compression",
    "computer_use",
    "data",
    "deployment",
    "eval",
    "federation",
    "inference",
    "longcontext",
    "mcp",
    "memory",
    "model",
    "monitoring",
    "multiagent",
    "multimodal",
    "protocol",
    "reasoning",
    "retrieval",
    "safety",
    "search",
    "serving",
    "simulation",
    "tools",
    "training",
    "ui",
)

_LEGACY_MODULES = (
    "api_registry",
    "agent_registry",
    "registry_snapshot",
    "skills_registry",
    "neural_brain",
    "self_upgrade",
)

__all__ = list(_PUBLIC_SUBPACKAGES + _LEGACY_MODULES)


def __getattr__(name: str):
    if name in _PUBLIC_SUBPACKAGES:
        module = import_module(f"src.{name}")
    elif name in _LEGACY_MODULES:
        module = import_module(name)
    elif os.path.exists(os.path.join(_REPO_ROOT, f"{name}.py")) or os.path.exists(
        os.path.join(_REPO_ROOT, name, "__init__.py")
    ):
        module = import_module(name)
    else:
        raise AttributeError(f"module 'aurelius' has no attribute {name!r}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC_SUBPACKAGES) | set(_LEGACY_MODULES))
