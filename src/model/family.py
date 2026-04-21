"""Model family + variant registry.

Sits on top of ``src.model.manifest`` — wraps ``FamilyManifest`` objects in
``ModelVariant`` wrappers with human-readable metadata, groups them into
named ``ModelFamily`` objects, and exposes module-level registries keyed
by family name and by ``"{family}/{variant}"`` variant id.

Pure stdlib. Infrastructure-only: introduces no new config flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .manifest import FamilyManifest as _FamilyManifest

# Defensive import: sibling manifest.py may be built in parallel with this
# module. At production import time the manifest module exists, so the
# normal path succeeds; during certain test orderings it may not, and we
# fall back to a permissive placeholder so this module still loads.
try:
    from .manifest import (  # type: ignore[assignment]
        AURELIUS_REFERENCE_MANIFEST,
        FamilyManifest,
    )

    _MANIFEST_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when manifest absent
    FamilyManifest = Any  # type: ignore[assignment,misc]
    AURELIUS_REFERENCE_MANIFEST = None  # type: ignore[assignment]
    _MANIFEST_AVAILABLE = False


@dataclass(frozen=True)
class ModelVariant:
    """A named, human-documented wrapper around a :class:`FamilyManifest`."""

    manifest: "_FamilyManifest"
    description: str
    release_notes: str = ""

    def __post_init__(self) -> None:
        if self.manifest is None:
            raise TypeError(
                "ModelVariant.manifest is required (got None); "
                "pass a FamilyManifest instance."
            )


@dataclass
class ModelFamily:
    """A named collection of :class:`ModelVariant` members."""

    family_name: str
    variants: dict[str, ModelVariant] = field(default_factory=dict)
    default_variant: str | None = None

    def add_variant(self, name: str, variant: ModelVariant) -> None:
        """Register ``variant`` under ``name``; reject duplicate names."""
        if name in self.variants:
            raise ValueError(
                f"variant {name!r} already registered in family "
                f"{self.family_name!r}"
            )
        self.variants[name] = variant

    def get_variant(self, name: str) -> ModelVariant:
        """Return variant ``name`` or raise ``KeyError``."""
        if name not in self.variants:
            raise KeyError(
                f"variant {name!r} not found in family {self.family_name!r}; "
                f"known variants: {sorted(self.variants)}"
            )
        return self.variants[name]

    def list_variants(self) -> tuple[str, ...]:
        """Return variant names in deterministic (insertion) order."""
        return tuple(self.variants.keys())


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

MODEL_FAMILY_REGISTRY: dict[str, ModelFamily] = {}
MODEL_VARIANT_REGISTRY: dict[str, ModelVariant] = {}


def register_family(family: ModelFamily) -> ModelFamily:
    """Insert ``family`` into :data:`MODEL_FAMILY_REGISTRY`.

    Also back-fills every contained variant into
    :data:`MODEL_VARIANT_REGISTRY` under ``"{family_name}/{variant_name}"``.
    """
    if family.family_name in MODEL_FAMILY_REGISTRY:
        raise ValueError(
            f"family {family.family_name!r} already registered"
        )
    MODEL_FAMILY_REGISTRY[family.family_name] = family
    for variant_name, variant in family.variants.items():
        variant_id = f"{family.family_name}/{variant_name}"
        MODEL_VARIANT_REGISTRY[variant_id] = variant
    return family


def register_variant(
    family_name: str, variant_name: str, variant: ModelVariant
) -> str:
    """Register ``variant`` into an existing family.

    Returns the fully qualified variant id ``"{family}/{variant}"``.
    """
    if family_name not in MODEL_FAMILY_REGISTRY:
        raise KeyError(
            f"family {family_name!r} not registered; "
            f"call register_family() first"
        )
    family = MODEL_FAMILY_REGISTRY[family_name]
    family.add_variant(variant_name, variant)
    variant_id = f"{family_name}/{variant_name}"
    MODEL_VARIANT_REGISTRY[variant_id] = variant
    return variant_id


def get_family(name: str) -> ModelFamily:
    """Return family ``name`` or raise ``KeyError``."""
    if name not in MODEL_FAMILY_REGISTRY:
        raise KeyError(
            f"family {name!r} not registered; "
            f"known families: {sorted(MODEL_FAMILY_REGISTRY)}"
        )
    return MODEL_FAMILY_REGISTRY[name]


def get_variant_by_id(variant_id: str) -> ModelVariant:
    """Return variant by ``"{family}/{variant}"`` id or raise ``KeyError``."""
    if variant_id not in MODEL_VARIANT_REGISTRY:
        raise KeyError(
            f"variant id {variant_id!r} not registered; "
            f"known variants: {sorted(MODEL_VARIANT_REGISTRY)}"
        )
    return MODEL_VARIANT_REGISTRY[variant_id]


# ---------------------------------------------------------------------------
# Reference family auto-registration
# ---------------------------------------------------------------------------

AURELIUS_FAMILY: ModelFamily | None = None

if _MANIFEST_AVAILABLE and AURELIUS_REFERENCE_MANIFEST is not None:
    _base_variant = ModelVariant(
        manifest=AURELIUS_REFERENCE_MANIFEST,
        description=(
            "Aurelius reference 1.395B-parameter decoder-only transformer "
            "(baseline configuration)."
        ),
        release_notes="Cycle-122 baseline reference variant.",
    )
    AURELIUS_FAMILY = ModelFamily(
        family_name="aurelius",
        variants={"base-1.395b": _base_variant},
        default_variant="base-1.395b",
    )
    register_family(AURELIUS_FAMILY)


__all__ = [
    "AURELIUS_FAMILY",
    "MODEL_FAMILY_REGISTRY",
    "MODEL_VARIANT_REGISTRY",
    "ModelFamily",
    "ModelVariant",
    "get_family",
    "get_variant_by_id",
    "register_family",
    "register_variant",
]
