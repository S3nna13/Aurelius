"""Backend-aware factory bridge.

This module provides a thin wrapper around the model factory that resolves
the appropriate :class:`BackendAdapter` for a given manifest and returns
both the adapter and the downstream factory result bundled together.

Using a bridge -- rather than modifying ``src/model/factory.py`` directly
-- keeps the model factory free of any backend-surface coupling and
preserves the frozen-file contract.

No foreign imports are permitted in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.backends.base import BackendAdapter, BackendAdapterError
from src.backends.registry import select_backend_for_manifest

__all__ = [
    "BackendBuildResult",
    "build_with_backend",
]


@dataclass(frozen=True)
class BackendBuildResult:
    """Bundle returned by :func:`build_with_backend`.

    ``adapter`` is the resolved :class:`BackendAdapter`. ``backbone`` is
    the object produced by the downstream model factory (or ``None`` if
    the factory call was skipped, e.g. during a dry lookup).
    """

    adapter: BackendAdapter
    backbone: Any


def build_with_backend(
    manifest: Any,
    *,
    build_backbone: bool = False,
    aurelius_config: Any | None = None,
) -> BackendBuildResult:
    """Resolve a backend adapter for ``manifest`` and optionally build a backbone.

    By default this performs backend resolution only and returns
    ``backbone=None``. Callers that want the backbone materialized should
    pass ``build_backbone=True``; the model factory is imported lazily so
    this bridge remains cheap to load in pure-adapter workflows.
    """
    # Lazy import so FamilyManifest's existence is only required inside
    # select_backend_for_manifest, and so the model factory is not pulled
    # in for callers that only want adapter resolution.
    from src.model.manifest import FamilyManifest

    if not isinstance(manifest, FamilyManifest):
        raise BackendAdapterError(
            f"build_with_backend expected a FamilyManifest, got "
            f"{type(manifest).__name__}"
        )

    adapter = select_backend_for_manifest(manifest)
    backbone: Any = None
    if build_backbone:
        from src.model.factory import build_backbone_from_manifest

        backbone = build_backbone_from_manifest(
            manifest, aurelius_config=aurelius_config
        )
    return BackendBuildResult(adapter=adapter, backbone=backbone)
