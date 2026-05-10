"""Architecture registry — all architectures registered here for discovery."""

from __future__ import annotations

ARCHITECTURE_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    ARCHITECTURE_REGISTRY[name] = cls


def list_architectures(category: str | None = None) -> list[str]:
    if category is None:
        return list(ARCHITECTURE_REGISTRY.keys())
    return [k for k in ARCHITECTURE_REGISTRY if k.startswith(category)]


def get_architecture(name: str) -> type | None:
    return ARCHITECTURE_REGISTRY.get(name)
