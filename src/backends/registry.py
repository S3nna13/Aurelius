"""Module-level registries for Aurelius backends and engine adapters.

This module defines two dict registries and the small set of helper
functions that mediate all reads/writes:

* ``BACKEND_REGISTRY`` -- backend-name -> :class:`BackendAdapter`
* ``ENGINE_ADAPTER_REGISTRY`` -- engine-name -> :class:`EngineAdapter`

Both registries are plain ``dict`` objects so they remain inspectable and
mutable under test. All lookups go through typed accessors that raise
:class:`BackendAdapterError` on lookup failure; there are no silent
fallbacks.

The manifest-aware :func:`select_backend_for_manifest` uses a lazy import
of :class:`FamilyManifest` so this module never pulls in ``src.model`` at
import time -- the registry surface must stay cheap to load.

No foreign imports are permitted in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    EngineAdapter,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "BACKEND_REGISTRY",
    "ENGINE_ADAPTER_REGISTRY",
    "register_backend",
    "get_backend",
    "list_backends",
    "register_engine_adapter",
    "get_engine_adapter",
    "list_engine_adapters",
    "select_backend_for_manifest",
]


BACKEND_REGISTRY: dict[str, BackendAdapter] = {}
ENGINE_ADAPTER_REGISTRY: dict[str, EngineAdapter] = {}


def register_backend(
    adapter: BackendAdapter,
    *,
    overwrite: bool = False,
) -> None:
    """Register a :class:`BackendAdapter` under ``adapter.contract.backend_name``.

    Raises :class:`BackendAdapterError` if ``adapter`` is not a
    :class:`BackendAdapter` instance, or if a different adapter is already
    registered under the same name and ``overwrite`` is False.
    """
    if not isinstance(adapter, BackendAdapter):
        raise BackendAdapterError(
            f"register_backend expected a BackendAdapter instance, got {type(adapter).__name__}"
        )
    name = adapter.contract.backend_name
    if name in BACKEND_REGISTRY and not overwrite:
        raise BackendAdapterError(
            f"backend already registered: {name!r} (pass overwrite=True to replace)"
        )
    BACKEND_REGISTRY[name] = adapter


def get_backend(name: str) -> BackendAdapter:
    """Return the registered backend adapter named ``name``."""
    if not isinstance(name, str) or not name:
        raise BackendAdapterError(f"backend name must be a non-empty string, got {name!r}")
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(BACKEND_REGISTRY.keys())) or "<none>"
        raise BackendAdapterError(f"backend not registered: {name!r} (known: {known})") from exc


def list_backends() -> tuple[str, ...]:
    """Return a stable-sorted tuple of registered backend names."""
    return tuple(sorted(BACKEND_REGISTRY.keys()))


def register_engine_adapter(
    adapter: EngineAdapter,
    *,
    overwrite: bool = False,
) -> None:
    """Register an :class:`EngineAdapter` under its contract's backend_name."""
    if not isinstance(adapter, EngineAdapter):
        raise BackendAdapterError(
            f"register_engine_adapter expected an EngineAdapter instance, got "
            f"{type(adapter).__name__}"
        )
    name = adapter.contract.backend_name
    if name in ENGINE_ADAPTER_REGISTRY and not overwrite:
        raise BackendAdapterError(
            f"engine adapter already registered: {name!r} (pass overwrite=True to replace)"
        )
    ENGINE_ADAPTER_REGISTRY[name] = adapter


def get_engine_adapter(name: str) -> EngineAdapter:
    """Return the registered engine adapter named ``name``."""
    if not isinstance(name, str) or not name:
        raise BackendAdapterError(f"engine adapter name must be a non-empty string, got {name!r}")
    try:
        return ENGINE_ADAPTER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(ENGINE_ADAPTER_REGISTRY.keys())) or "<none>"
        raise BackendAdapterError(
            f"engine adapter not registered: {name!r} (known: {known})"
        ) from exc


def list_engine_adapters() -> tuple[str, ...]:
    """Return a stable-sorted tuple of registered engine adapter names."""
    return tuple(sorted(ENGINE_ADAPTER_REGISTRY.keys()))


def select_backend_for_manifest(manifest: Any) -> BackendAdapter:
    """Return the backend adapter that should run ``manifest``.

    Uses a lazy import of :class:`FamilyManifest` so this module does not
    pull :mod:`src.model` at registry-load time. If the manifest has no
    ``backend_name`` set, the v6 default ``"pytorch"`` is returned.
    """
    from src.model.manifest import FamilyManifest

    if not isinstance(manifest, FamilyManifest):
        raise BackendAdapterError(
            f"select_backend_for_manifest expected a FamilyManifest, got {type(manifest).__name__}"
        )

    name = manifest.backend_name
    if name is None:
        if "pytorch" not in BACKEND_REGISTRY:
            raise BackendAdapterError(
                "manifest has no backend_name and the v6 default 'pytorch' "
                "is not registered; register a pytorch adapter or set "
                "manifest.backend_name explicitly"
            )
        return BACKEND_REGISTRY["pytorch"]

    if name not in BACKEND_REGISTRY:
        known = ", ".join(sorted(BACKEND_REGISTRY.keys())) or "<none>"
        raise BackendAdapterError(
            f"manifest requests backend {name!r} but it is not registered "
            f"(known: {known}); register the adapter before building"
        )
    return BACKEND_REGISTRY[name]
