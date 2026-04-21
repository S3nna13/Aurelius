"""Family-aware model factory.

Materializes a backbone from a :class:`FamilyManifest` or a variant id.
The factory resolves ``manifest.backbone_class`` via :mod:`importlib`,
optionally routed through a registered builder override.

Pure stdlib: dataclasses, importlib, typing.

Usage::

    from src.model.factory import build_from_variant_id
    model = build_from_variant_id("aurelius/base-1.395b")

For test isolation, the factory also ships a lightweight
``_DummyBackbone`` class that can be referenced by a stub manifest via
``src.model.factory._DummyBackbone``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.model.manifest import FamilyManifest

__all__ = [
    "FactoryError",
    "build_backbone_from_manifest",
    "build_from_variant_id",
    "register_backbone_builder",
    "DEFAULT_BACKBONE_BUILDERS",
    "_DummyBackbone",
]


class FactoryError(Exception):
    """Raised when a backbone cannot be materialized from a manifest."""


@dataclass
class _DummyBackbone:
    """Lightweight stand-in used by tests and by factory smoke checks.

    Not a real model. Stores the shape-level hyperparameters that the
    factory would pass to any concrete backbone. Instantiation is cheap
    and side-effect-free, which makes it safe to use in unit tests that
    should not touch PyTorch or allocate tensors.
    """

    vocab_size: int
    max_seq_len: int
    d_model: int = 64
    n_layers: int = 2
    variant_name: str = ""
    family_name: str = ""
    testing_only: bool = True


def _resolve_class(dotted_path: str) -> type:
    """Resolve ``module.sub.ClassName`` via :mod:`importlib`."""
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise FactoryError(
            f"backbone_class must be a dotted path 'module.ClassName', "
            f"got {dotted_path!r}"
        )
    module_path, _, attr = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise FactoryError(
            f"could not import module {module_path!r} for backbone_class "
            f"{dotted_path!r}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise FactoryError(
            f"module {module_path!r} has no attribute {attr!r} "
            f"(from backbone_class {dotted_path!r})"
        )
    return getattr(module, attr)


def _default_aurelius_builder(
    backbone_cls: type,
    manifest: "FamilyManifest",
    aurelius_config: Any | None,
) -> Any:
    """Default builder for :class:`AureliusTransformer`-shaped classes.

    When ``aurelius_config`` is None we defer to the class's own default
    construction path (which uses tiny defaults under the hood), keeping
    tests cheap and avoiding 1.4B-parameter allocations.
    """
    if aurelius_config is None:
        return backbone_cls()
    return backbone_cls(aurelius_config)


def _dummy_builder(
    backbone_cls: type,
    manifest: "FamilyManifest",
    aurelius_config: Any | None,
) -> Any:
    """Builder for the in-module :class:`_DummyBackbone` stub."""
    kwargs: dict[str, Any] = dict(
        vocab_size=manifest.vocab_size,
        max_seq_len=manifest.max_seq_len,
        variant_name=manifest.variant_name,
        family_name=manifest.family_name,
    )
    if aurelius_config is not None:
        # Pull shape-level hints if the config exposes them.
        for attr in ("d_model", "n_layers"):
            if hasattr(aurelius_config, attr):
                kwargs[attr] = getattr(aurelius_config, attr)
    return backbone_cls(**kwargs)


# Module-level registry of dotted-path -> builder callable.
# Signature: builder(backbone_cls, manifest, aurelius_config) -> Any.
DEFAULT_BACKBONE_BUILDERS: dict[str, Callable[..., Any]] = {
    "src.model.transformer.AureliusTransformer": _default_aurelius_builder,
    "src.model.factory._DummyBackbone": _dummy_builder,
}


def register_backbone_builder(
    backbone_class_path: str,
    builder_fn: Callable[..., Any],
) -> None:
    """Register (or override) a builder for a given dotted backbone path.

    Raises :class:`FactoryError` if inputs are invalid. Existing entries
    are overwritten silently, which is intentional: callers use this to
    override the factory behavior for a specific family variant.
    """
    if not isinstance(backbone_class_path, str) or not backbone_class_path:
        raise FactoryError(
            f"backbone_class_path must be a non-empty string, got "
            f"{backbone_class_path!r}"
        )
    if not callable(builder_fn):
        raise FactoryError(
            f"builder_fn must be callable, got {type(builder_fn).__name__}"
        )
    DEFAULT_BACKBONE_BUILDERS[backbone_class_path] = builder_fn


def build_backbone_from_manifest(
    manifest: "FamilyManifest",
    aurelius_config: Any | None = None,
) -> Any:
    """Instantiate the backbone class named by ``manifest.backbone_class``.

    Looks up an override in :data:`DEFAULT_BACKBONE_BUILDERS` first; if
    none is registered, falls back to calling the class with either
    ``(aurelius_config,)`` or ``()`` depending on whether a config is
    supplied.
    """
    # Lazy import to avoid a hard coupling to manifest.py at module load.
    from src.model.manifest import FamilyManifest

    if not isinstance(manifest, FamilyManifest):
        raise FactoryError(
            f"manifest must be a FamilyManifest, got {type(manifest).__name__}"
        )

    dotted = manifest.backbone_class
    if not isinstance(dotted, str) or not dotted:
        raise FactoryError(
            f"manifest.backbone_class must be a non-empty dotted path, got "
            f"{dotted!r}"
        )

    backbone_cls = _resolve_class(dotted)
    builder = DEFAULT_BACKBONE_BUILDERS.get(dotted)
    if builder is None:
        # Generic fallback: call with or without config.
        try:
            if aurelius_config is None:
                return backbone_cls()
            return backbone_cls(aurelius_config)
        except TypeError as exc:
            raise FactoryError(
                f"generic builder failed for {dotted!r}: {exc}"
            ) from exc

    try:
        return builder(backbone_cls, manifest, aurelius_config)
    except FactoryError:
        raise
    except Exception as exc:
        raise FactoryError(
            f"builder for {dotted!r} raised: {exc}"
        ) from exc


def build_from_variant_id(
    variant_id: str,
    aurelius_config: Any | None = None,
) -> Any:
    """Look up ``variant_id`` -> manifest, then build the backbone.

    ``variant_id`` is ``"family_name/variant_name"`` (the same key used by
    :data:`src.model.manifest.MODEL_MANIFEST_REGISTRY`). The lookup first
    consults the sibling variant registry (``src.model.family``) if
    present; otherwise it falls back directly to the manifest registry.
    """
    if not isinstance(variant_id, str) or not variant_id:
        raise FactoryError(
            f"variant_id must be a non-empty string, got {variant_id!r}"
        )

    manifest = _lookup_manifest_for_variant(variant_id)
    return build_backbone_from_manifest(manifest, aurelius_config=aurelius_config)


def _lookup_manifest_for_variant(variant_id: str) -> "FamilyManifest":
    """Resolve ``variant_id`` to a manifest.

    Prefers the variant registry from :mod:`src.model.family` when that
    module is importable; otherwise uses the manifest registry directly.
    """
    # Try the sibling family module first.
    try:
        family_mod = importlib.import_module("src.model.family")
    except ImportError:
        family_mod = None

    if family_mod is not None:
        registry = getattr(family_mod, "MODEL_VARIANT_REGISTRY", None)
        if isinstance(registry, dict) and variant_id in registry:
            variant = registry[variant_id]
            manifest = getattr(variant, "manifest", None)
            if manifest is not None:
                return manifest

    # Fall back to manifest registry.
    from src.model.manifest import MODEL_MANIFEST_REGISTRY

    if variant_id in MODEL_MANIFEST_REGISTRY:
        return MODEL_MANIFEST_REGISTRY[variant_id]

    raise FactoryError(f"unknown variant_id: {variant_id!r}")
