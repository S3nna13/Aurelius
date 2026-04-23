"""Family manifest for the Aurelius model family.

Under Meta-Prompt v5 (Model Family Edition), each family member publishes
a :class:`FamilyManifest` describing its identity, backbone class, tokenizer,
context policy, capability tags, and versioning metadata. Manifests replace
ad-hoc boolean flags on :class:`AureliusConfig` for family-level identity.

Siblings (not imported; referenced for context only):
    - ``src/model/family.py``        -- family enumeration / selection
    - ``src/model/factory.py``       -- build a backbone from a manifest
    - ``src/model/compatibility.py`` -- cross-version compatibility checks

Pure stdlib: dataclasses, enum, json, re, typing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping


__all__ = [
    "ReleaseTrack",
    "FamilyManifest",
    "ManifestValidationError",
    "load_manifest",
    "dump_manifest",
    "register_manifest",
    "get_manifest",
    "list_manifests",
    "MODEL_MANIFEST_REGISTRY",
    "AURELIUS_REFERENCE_MANIFEST",
]


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_BACKEND_NAME_RE = re.compile(r"^[a-z0-9_\-]+$")


class ManifestValidationError(Exception):
    """Raised when a manifest payload is malformed."""


class ReleaseTrack(str, Enum):
    """Release maturity track for a family variant."""

    RESEARCH = "research"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class FamilyManifest:
    """Canonical identity + versioning record for a family variant."""

    family_name: str
    variant_name: str
    backbone_class: str
    tokenizer_name: str
    tokenizer_hash: str | None
    vocab_size: int
    max_seq_len: int
    context_policy: str
    rope_config: dict
    capability_tags: tuple[str, ...]
    checkpoint_format_version: str
    config_version: str
    compatibility_version: str
    release_track: str
    migration_notes: tuple[str, ...] = field(default_factory=tuple)
    backend_name: str | None = None
    engine_contract: str | None = None
    adapter_contract: str | None = None

    def __post_init__(self) -> None:
        # String identity fields
        for fname in ("family_name", "variant_name", "backbone_class",
                      "tokenizer_name", "context_policy"):
            v = getattr(self, fname)
            if not isinstance(v, str) or not v:
                raise ManifestValidationError(
                    f"{fname} must be a non-empty string, got {v!r}"
                )

        if self.tokenizer_hash is not None and not isinstance(self.tokenizer_hash, str):
            raise ManifestValidationError(
                f"tokenizer_hash must be str or None, got {type(self.tokenizer_hash).__name__}"
            )

        if not isinstance(self.vocab_size, int) or isinstance(self.vocab_size, bool) \
                or self.vocab_size <= 0:
            raise ManifestValidationError(
                f"vocab_size must be a positive int, got {self.vocab_size!r}"
            )
        if not isinstance(self.max_seq_len, int) or isinstance(self.max_seq_len, bool) \
                or self.max_seq_len <= 0:
            raise ManifestValidationError(
                f"max_seq_len must be a positive int, got {self.max_seq_len!r}"
            )

        if not isinstance(self.rope_config, dict):
            raise ManifestValidationError(
                f"rope_config must be a dict, got {type(self.rope_config).__name__}"
            )

        # Normalize capability_tags / migration_notes to tuples of strings.
        object.__setattr__(self, "capability_tags",
                           _coerce_str_tuple(self.capability_tags, "capability_tags"))
        object.__setattr__(self, "migration_notes",
                           _coerce_str_tuple(self.migration_notes, "migration_notes"))

        for vfield in ("checkpoint_format_version", "config_version",
                       "compatibility_version"):
            v = getattr(self, vfield)
            if not isinstance(v, str) or not _SEMVER_RE.match(v):
                raise ManifestValidationError(
                    f"{vfield} must match semver X.Y.Z, got {v!r}"
                )

        try:
            ReleaseTrack(self.release_track)
        except ValueError as exc:
            raise ManifestValidationError(
                f"release_track {self.release_track!r} not in "
                f"{[t.value for t in ReleaseTrack]}"
            ) from exc

        if self.backend_name is not None:
            if not isinstance(self.backend_name, str) or not self.backend_name:
                raise ManifestValidationError(
                    f"backend_name must be a non-empty string or None, "
                    f"got {self.backend_name!r}"
                )
            if not _BACKEND_NAME_RE.match(self.backend_name):
                raise ManifestValidationError(
                    f"backend_name must match [a-z0-9_-]+ (lower-snake-or-dash), "
                    f"got {self.backend_name!r}"
                )

        for cfield in ("engine_contract", "adapter_contract"):
            v = getattr(self, cfield)
            if v is None:
                continue
            if not isinstance(v, str) or not _SEMVER_RE.match(v):
                raise ManifestValidationError(
                    f"{cfield} must match semver X.Y.Z or be None, got {v!r}"
                )

    @property
    def registry_key(self) -> str:
        return f"{self.family_name}/{self.variant_name}"


def _coerce_str_tuple(value: Any, fname: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ManifestValidationError(
            f"{fname} must be a sequence of strings, got bare str"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ManifestValidationError(
            f"{fname} must be iterable, got {type(value).__name__}"
        ) from exc
    for item in items:
        if not isinstance(item, str):
            raise ManifestValidationError(
                f"{fname} entries must be str, got {type(item).__name__}"
            )
    return items


_MANIFEST_FIELDS = tuple(f.name for f in fields(FamilyManifest))
_OPTIONAL_MANIFEST_FIELDS = frozenset({
    "migration_notes",
    "backend_name",
    "engine_contract",
    "adapter_contract",
})


def load_manifest(data: Mapping[str, Any]) -> FamilyManifest:
    """Validate a mapping and construct a :class:`FamilyManifest`."""
    if not isinstance(data, Mapping):
        raise ManifestValidationError(
            f"manifest data must be a mapping, got {type(data).__name__}"
        )
    missing = [f for f in _MANIFEST_FIELDS
               if f not in data and f not in _OPTIONAL_MANIFEST_FIELDS]
    if missing:
        raise ManifestValidationError(f"manifest missing fields: {missing}")
    extra = [k for k in data.keys() if k not in _MANIFEST_FIELDS]
    if extra:
        raise ManifestValidationError(f"manifest has unknown fields: {extra}")

    kwargs: dict[str, Any] = {}
    for fname in _MANIFEST_FIELDS:
        if fname in data:
            kwargs[fname] = data[fname]
    # rope_config: copy so callers can't mutate the stored dict via input ref.
    if isinstance(kwargs.get("rope_config"), dict):
        kwargs["rope_config"] = dict(kwargs["rope_config"])
    return FamilyManifest(**kwargs)


def dump_manifest(manifest: FamilyManifest) -> dict:
    """Serialize a manifest to a JSON-safe dict."""
    if not isinstance(manifest, FamilyManifest):
        raise ManifestValidationError(
            f"dump_manifest expected FamilyManifest, got {type(manifest).__name__}"
        )
    return {
        "family_name": manifest.family_name,
        "variant_name": manifest.variant_name,
        "backbone_class": manifest.backbone_class,
        "tokenizer_name": manifest.tokenizer_name,
        "tokenizer_hash": manifest.tokenizer_hash,
        "vocab_size": manifest.vocab_size,
        "max_seq_len": manifest.max_seq_len,
        "context_policy": manifest.context_policy,
        "rope_config": dict(manifest.rope_config),
        "capability_tags": list(manifest.capability_tags),
        "checkpoint_format_version": manifest.checkpoint_format_version,
        "config_version": manifest.config_version,
        "compatibility_version": manifest.compatibility_version,
        "release_track": manifest.release_track,
        "migration_notes": list(manifest.migration_notes),
        "backend_name": manifest.backend_name,
        "engine_contract": manifest.engine_contract,
        "adapter_contract": manifest.adapter_contract,
    }


MODEL_MANIFEST_REGISTRY: dict[str, FamilyManifest] = {}


def register_manifest(manifest: FamilyManifest) -> None:
    """Register a manifest in the module-level registry.

    Raises :class:`ManifestValidationError` if ``family/variant`` is already
    registered (duplicate registration is a programming error).
    """
    if not isinstance(manifest, FamilyManifest):
        raise ManifestValidationError(
            f"register_manifest expected FamilyManifest, got {type(manifest).__name__}"
        )
    key = manifest.registry_key
    if key in MODEL_MANIFEST_REGISTRY:
        raise ManifestValidationError(f"manifest already registered: {key}")
    MODEL_MANIFEST_REGISTRY[key] = manifest


def get_manifest(family: str, variant: str) -> FamilyManifest:
    """Return a registered manifest by family/variant."""
    key = f"{family}/{variant}"
    try:
        return MODEL_MANIFEST_REGISTRY[key]
    except KeyError as exc:
        raise ManifestValidationError(f"manifest not registered: {key}") from exc


def list_manifests() -> tuple[FamilyManifest, ...]:
    """Return all registered manifests (stable iteration order)."""
    return tuple(MODEL_MANIFEST_REGISTRY.values())


# ---------------------------------------------------------------------------
# Reference manifest: the 1.395B Aurelius backbone.
# ---------------------------------------------------------------------------

AURELIUS_REFERENCE_MANIFEST: FamilyManifest = FamilyManifest(
    family_name="aurelius",
    variant_name="base-1.395b",
    backbone_class="src.model.transformer.AureliusTransformer",
    tokenizer_name="aurelius-bpe",
    tokenizer_hash=None,
    vocab_size=128000,
    max_seq_len=8192,
    context_policy="rope_yarn",
    rope_config={"theta": 500000, "yarn_scale": 1.0},
    capability_tags=("base",),
    checkpoint_format_version="1.0.0",
    config_version="1.0.0",
    compatibility_version="1.0.0",
    release_track="research",
    migration_notes=("initial reference manifest",),
    backend_name="pytorch",
    engine_contract="1.0.0",
    adapter_contract="1.0.0",
)

# Auto-register at import time. JSON-safety check is advisory: dump_manifest
# is guaranteed JSON-safe by construction, but we assert once here to fail
# fast if the reference ever drifts.
json.dumps(dump_manifest(AURELIUS_REFERENCE_MANIFEST))
if AURELIUS_REFERENCE_MANIFEST.registry_key not in MODEL_MANIFEST_REGISTRY:
    register_manifest(AURELIUS_REFERENCE_MANIFEST)
