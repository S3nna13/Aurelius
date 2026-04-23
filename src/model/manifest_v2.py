"""Manifest schema v2 helpers for the Aurelius model family.

Meta-Prompt v6 extends the family manifest with three backend-identity fields:

    - ``backend_name``     -- e.g. ``"pytorch"``, ``"jax"``, ``"vllm"``, ...
    - ``engine_contract``  -- semver ``X.Y.Z`` of the engine/adapter protocol
    - ``adapter_contract`` -- semver ``X.Y.Z`` of the variant-adapter protocol

These fields live on :class:`src.model.manifest.FamilyManifest` as additive,
nullable fields so every pre-v2 manifest remains legal. A manifest is "v2"
iff all three new fields are populated (not ``None``). This module provides:

    - a :data:`MANIFEST_SCHEMA_VERSION` constant for v2,
    - :func:`is_v2_manifest` predicate,
    - :func:`upgrade_to_v2` constructor that returns a new
      :class:`FamilyManifest` with the three new fields populated,
    - :func:`v2_to_v1_dict` for legacy-compatible serialization,
    - :func:`compare_backend_contracts` for backend-aware compatibility,
    - :func:`list_v2_manifests` lazy reader over the single source of truth
      registry ``MODEL_MANIFEST_REGISTRY``.

The severity verdict strings returned by :func:`compare_backend_contracts`
are deliberately identical to the constants exported by
``src.model.compatibility`` (``"exact"``, ``"minor_mismatch"``,
``"major_break"``) so callers can treat them interchangeably. This module
parses semver locally to stay independent of ``compatibility.py``.

Siblings (not imported; referenced for context only):
    - ``src/model/manifest.py``      -- source of truth for FamilyManifest
    - ``src/model/compatibility.py`` -- cross-version compatibility checks

Pure stdlib: dataclasses, re, typing.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from .manifest import (
    MODEL_MANIFEST_REGISTRY,
    FamilyManifest,
    ManifestValidationError,
    dump_manifest,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "is_v2_manifest",
    "upgrade_to_v2",
    "v2_to_v1_dict",
    "compare_backend_contracts",
    "list_v2_manifests",
]


MANIFEST_SCHEMA_VERSION: str = "2.0.0"

_V2_FIELDS: tuple[str, ...] = (
    "backend_name",
    "engine_contract",
    "adapter_contract",
)

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_BACKEND_NAME_RE = re.compile(r"^[a-z0-9_\-]+$")

# Verdict constants -- MUST stay identical to src.model.compatibility.
_VERDICT_EXACT = "exact"
_VERDICT_MINOR = "minor_mismatch"
_VERDICT_MAJOR = "major_break"


def is_v2_manifest(manifest: FamilyManifest) -> bool:
    """Return ``True`` iff all three backend-identity fields are populated.

    A "v2" manifest has ``backend_name``, ``engine_contract``, and
    ``adapter_contract`` all non-``None``. Pre-v2 (aka v1) manifests may
    have any or none of these fields set to ``None``.

    Args:
        manifest: Any :class:`FamilyManifest`.

    Returns:
        True if all three v2 fields are non-``None``.

    Raises:
        ManifestValidationError: if ``manifest`` is not a FamilyManifest.
    """
    if not isinstance(manifest, FamilyManifest):
        raise ManifestValidationError(
            f"is_v2_manifest expected FamilyManifest, got "
            f"{type(manifest).__name__}"
        )
    return all(getattr(manifest, f) is not None for f in _V2_FIELDS)


def upgrade_to_v2(
    manifest: FamilyManifest,
    *,
    backend_name: str,
    engine_contract: str = "1.0.0",
    adapter_contract: str = "1.0.0",
) -> FamilyManifest:
    """Return a new :class:`FamilyManifest` with the three v2 fields populated.

    The input manifest is left unchanged (FamilyManifest is a frozen
    dataclass). ``backend_name`` must match ``[a-z0-9_\\-]+``;
    ``engine_contract`` and ``adapter_contract`` must match semver
    ``X.Y.Z``. Validation is performed both here (to produce a clear error
    pre-construction) and again in :meth:`FamilyManifest.__post_init__`.

    Args:
        manifest:         Source v1 (or already-v2) manifest.
        backend_name:     Backend identifier, lower-snake-or-dash.
        engine_contract:  Engine/adapter protocol semver. Defaults to ``1.0.0``.
        adapter_contract: Variant-adapter protocol semver. Defaults to ``1.0.0``.

    Returns:
        A new FamilyManifest with the three v2 fields set.

    Raises:
        ManifestValidationError: on bad input types or malformed semver /
        backend_name.
    """
    if not isinstance(manifest, FamilyManifest):
        raise ManifestValidationError(
            f"upgrade_to_v2 expected FamilyManifest, got "
            f"{type(manifest).__name__}"
        )
    if not isinstance(backend_name, str) or not backend_name:
        raise ManifestValidationError(
            f"backend_name must be a non-empty string, got {backend_name!r}"
        )
    if not _BACKEND_NAME_RE.match(backend_name):
        raise ManifestValidationError(
            f"backend_name must match [a-z0-9_-]+ (lower-snake-or-dash), "
            f"got {backend_name!r}"
        )
    for cname, cval in (("engine_contract", engine_contract),
                        ("adapter_contract", adapter_contract)):
        if not isinstance(cval, str) or not _SEMVER_RE.match(cval):
            raise ManifestValidationError(
                f"{cname} must match semver X.Y.Z, got {cval!r}"
            )

    return dataclasses.replace(
        manifest,
        backend_name=backend_name,
        engine_contract=engine_contract,
        adapter_contract=adapter_contract,
    )


def v2_to_v1_dict(manifest: FamilyManifest) -> dict:
    """Dump a manifest as a v1-compatible dict.

    The three v2 keys (``backend_name``, ``engine_contract``,
    ``adapter_contract``) are stripped. The result is JSON-safe and can be
    consumed by legacy readers that don't know about schema v2.

    Args:
        manifest: Any :class:`FamilyManifest` (v1 or v2).

    Returns:
        A dict containing all v1 keys, in v1 order, with no v2 keys.

    Raises:
        ManifestValidationError: if ``manifest`` is not a FamilyManifest.
    """
    if not isinstance(manifest, FamilyManifest):
        raise ManifestValidationError(
            f"v2_to_v1_dict expected FamilyManifest, got "
            f"{type(manifest).__name__}"
        )
    payload = dump_manifest(manifest)
    for key in _V2_FIELDS:
        payload.pop(key, None)
    return payload


def _parse_semver_local(s: Any) -> tuple[int, int, int]:
    """Local, dependency-free semver parser.

    Kept private to this module so manifest_v2 does not import from
    ``src.model.compatibility``. Raises :class:`ManifestValidationError`
    rather than CompatibilityError because this module only deals with
    manifest-shaped data.
    """
    if not isinstance(s, str):
        raise ManifestValidationError(
            f"semver must be a string, got {type(s).__name__}"
        )
    m = _SEMVER_RE.match(s)
    if not m:
        raise ManifestValidationError(f"invalid semver: {s!r}")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def compare_backend_contracts(
    a: FamilyManifest,
    b: FamilyManifest,
) -> str:
    """Return a verdict string comparing two manifests' backend contracts.

    Rules, in order:

        1. Both ``backend_name is None``  -> ``"exact"`` (legacy v1 on both).
        2. Exactly one has ``backend_name``  -> ``"minor_mismatch"``.
        3. Both have ``backend_name`` but they differ  -> ``"major_break"``.
        4. Same ``backend_name``, ``engine_contract`` major differs
           -> ``"major_break"``.
        5. Same ``backend_name``, major matches but minor/patch differs
           -> ``"minor_mismatch"``.
        6. Exact match on all three  -> ``"exact"``.

    ``adapter_contract`` participates in rules (5)/(6) on equal footing
    with ``engine_contract`` for minor escalation: if any of the two
    contracts has a major mismatch, verdict is ``"major_break"``; if any
    has a minor/patch drift, verdict escalates to ``"minor_mismatch"``.

    Args:
        a, b: Two :class:`FamilyManifest` instances.

    Returns:
        One of ``"exact"``, ``"minor_mismatch"``, ``"major_break"``.

    Raises:
        ManifestValidationError: on bad inputs.
    """
    if not isinstance(a, FamilyManifest) or not isinstance(b, FamilyManifest):
        raise ManifestValidationError(
            "compare_backend_contracts expected two FamilyManifest instances"
        )

    a_name = a.backend_name
    b_name = b.backend_name

    if a_name is None and b_name is None:
        return _VERDICT_EXACT
    if (a_name is None) != (b_name is None):
        return _VERDICT_MINOR
    if a_name != b_name:
        return _VERDICT_MAJOR

    verdict = _VERDICT_EXACT
    for cname in ("engine_contract", "adapter_contract"):
        av = getattr(a, cname)
        bv = getattr(b, cname)
        if av is None and bv is None:
            continue
        if (av is None) != (bv is None):
            if verdict == _VERDICT_EXACT:
                verdict = _VERDICT_MINOR
            continue
        a_parts = _parse_semver_local(av)
        b_parts = _parse_semver_local(bv)
        if a_parts[0] != b_parts[0]:
            return _VERDICT_MAJOR
        if a_parts != b_parts and verdict == _VERDICT_EXACT:
            verdict = _VERDICT_MINOR
    return verdict


def list_v2_manifests() -> tuple[FamilyManifest, ...]:
    """Return the subset of registered manifests that are v2.

    Reads from ``MODEL_MANIFEST_REGISTRY`` lazily on every call so there
    is exactly one source of truth for registered manifests.

    Returns:
        A tuple of v2-qualifying manifests, in registry iteration order.
    """
    return tuple(m for m in MODEL_MANIFEST_REGISTRY.values() if is_v2_manifest(m))
