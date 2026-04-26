"""Public Aurelius package namespace.

This package provides a stable `aurelius.*` import path while preserving the
existing internal `src.*` package layout used throughout the repository.
"""

from __future__ import annotations

import sys
from importlib import import_module

__all__ = [
    "alignment",
    "data",
    "eval",
    "inference",
    "model",
    "serving",
    "training",
]


def _alias_legacy_package(package_name: str):
    module = import_module(f"src.{package_name}")
    sys.modules[f"{__name__}.{package_name}"] = module
    globals()[package_name] = module
    return module


for _package_name in __all__:
    _alias_legacy_package(_package_name)


del _package_name
