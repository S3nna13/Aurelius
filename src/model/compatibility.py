"""Version-compatibility gate for manifest / checkpoint / tokenizer.

Under Meta-Prompt v5 (Model Family Edition), the compatibility module
provides a well-defined verdict on whether two family manifests (or a
manifest and a checkpoint) can interoperate.

Compatibility severities:
    - ``exact``: no differences on any gated field.
    - ``minor_mismatch``: forward-compatible differences (e.g., yarn_scale,
      config_version minor drift). Callers are free to proceed with caution.
    - ``major_break``: incompatible; callers should refuse to load.

Pure stdlib: dataclasses, re, typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.model.manifest import FamilyManifest

__all__ = [
    "CompatibilityError",
    "SemverParts",
    "parse_semver",
    "CompatibilityVerdict",
    "check_manifest_compatibility",
    "check_checkpoint_compatibility",
    "assert_compatible",
]


_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Severity constants.
SEVERITY_EXACT = "exact"
SEVERITY_MINOR = "minor_mismatch"
SEVERITY_MAJOR = "major_break"

_SEVERITY_ORDER = {SEVERITY_EXACT: 0, SEVERITY_MINOR: 1, SEVERITY_MAJOR: 2}


class CompatibilityError(Exception):
    """Raised by :func:`assert_compatible` on a ``major_break`` verdict."""


@dataclass(frozen=True)
class SemverParts:
    """Parsed semver components."""

    major: int
    minor: int
    patch: int


def parse_semver(s: str) -> SemverParts:
    """Parse ``X.Y.Z`` into :class:`SemverParts`.

    Raises:
        CompatibilityError: if ``s`` is not a valid semver string.
    """
    if not isinstance(s, str):
        raise CompatibilityError(
            f"semver must be a string, got {type(s).__name__}"
        )
    match = _SEMVER_RE.match(s)
    if not match:
        raise CompatibilityError(f"invalid semver: {s!r}")
    return SemverParts(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
    )


@dataclass(frozen=True)
class CompatibilityVerdict:
    """Outcome of a compatibility check."""

    compatible: bool
    reasons: tuple[str, ...]
    severity: str

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_ORDER:
            raise CompatibilityError(
                f"invalid severity {self.severity!r}; must be one of "
                f"{sorted(_SEVERITY_ORDER)}"
            )
        if not isinstance(self.reasons, tuple):
            raise CompatibilityError(
                f"reasons must be a tuple, got {type(self.reasons).__name__}"
            )


def _escalate(current: str, new: str) -> str:
    """Return whichever severity is higher."""
    return new if _SEVERITY_ORDER[new] > _SEVERITY_ORDER[current] else current


def check_manifest_compatibility(
    required: "FamilyManifest",
    candidate: "FamilyManifest",
) -> CompatibilityVerdict:
    """Compare ``candidate`` against ``required`` and return a verdict.

    Rules:
        - major on ``compatibility_version`` must match exactly
          (otherwise ``major_break``); minor can differ forward-compatibly.
        - ``tokenizer_name`` must match exactly.
        - ``vocab_size`` must match exactly.
        - ``backbone_class`` must match exactly.
        - ``rope_config.theta`` must match; ``yarn_scale`` diffs are OK
          (downgraded to ``minor_mismatch``).
        - ``capability_tags``: candidate must be a superset of required.
    """
    reasons: list[str] = []
    severity = SEVERITY_EXACT

    # compatibility_version: major must match; minor drift is OK.
    req_v = parse_semver(required.compatibility_version)
    cand_v = parse_semver(candidate.compatibility_version)
    if req_v.major != cand_v.major:
        reasons.append(
            f"compatibility_version major mismatch: "
            f"required {required.compatibility_version!r}, "
            f"candidate {candidate.compatibility_version!r}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)
    elif req_v.minor != cand_v.minor or req_v.patch != cand_v.patch:
        reasons.append(
            f"compatibility_version minor/patch drift: "
            f"required {required.compatibility_version!r}, "
            f"candidate {candidate.compatibility_version!r}"
        )
        severity = _escalate(severity, SEVERITY_MINOR)

    if required.tokenizer_name != candidate.tokenizer_name:
        reasons.append(
            f"tokenizer_name mismatch: required {required.tokenizer_name!r}, "
            f"candidate {candidate.tokenizer_name!r}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)

    if required.vocab_size != candidate.vocab_size:
        reasons.append(
            f"vocab_size mismatch: required {required.vocab_size}, "
            f"candidate {candidate.vocab_size}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)

    if required.backbone_class != candidate.backbone_class:
        reasons.append(
            f"backbone_class mismatch: required {required.backbone_class!r}, "
            f"candidate {candidate.backbone_class!r}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)

    # rope_config.
    req_rope = required.rope_config or {}
    cand_rope = candidate.rope_config or {}
    if req_rope.get("theta") != cand_rope.get("theta"):
        reasons.append(
            f"rope_config.theta mismatch: required {req_rope.get('theta')!r}, "
            f"candidate {cand_rope.get('theta')!r}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)
    if req_rope.get("yarn_scale") != cand_rope.get("yarn_scale"):
        reasons.append(
            f"rope_config.yarn_scale differs: required "
            f"{req_rope.get('yarn_scale')!r}, candidate "
            f"{cand_rope.get('yarn_scale')!r}"
        )
        severity = _escalate(severity, SEVERITY_MINOR)

    # capability_tags: candidate must be a superset of required.
    req_tags = set(required.capability_tags)
    cand_tags = set(candidate.capability_tags)
    missing = req_tags - cand_tags
    if missing:
        reasons.append(
            f"capability_tags missing in candidate: {sorted(missing)}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)

    compatible = severity != SEVERITY_MAJOR
    return CompatibilityVerdict(
        compatible=compatible,
        reasons=tuple(reasons),
        severity=severity,
    )


def check_checkpoint_compatibility(
    manifest: "FamilyManifest",
    checkpoint_meta: dict,
) -> CompatibilityVerdict:
    """Check a checkpoint-meta dict against a manifest.

    ``checkpoint_meta`` must be a dict containing any subset of:
        - ``checkpoint_format_version`` (semver, required for gating)
        - ``config_version`` (semver, required for gating)
        - ``tokenizer_hash`` (str or None)

    Rules:
        - ``checkpoint_format_version`` major must match the manifest's.
        - ``config_version`` minor may drift (escalates to ``minor_mismatch``).
        - ``tokenizer_hash`` must match if both sides are non-None;
          if either is None, compatibility is permissive.
    """
    if not isinstance(checkpoint_meta, dict):
        raise CompatibilityError(
            f"checkpoint_meta must be a dict, got "
            f"{type(checkpoint_meta).__name__}"
        )

    reasons: list[str] = []
    severity = SEVERITY_EXACT

    ckpt_fmt = checkpoint_meta.get("checkpoint_format_version")
    if ckpt_fmt is None:
        reasons.append("checkpoint_format_version missing in checkpoint_meta")
        severity = _escalate(severity, SEVERITY_MAJOR)
    else:
        man_fmt = parse_semver(manifest.checkpoint_format_version)
        cand_fmt = parse_semver(ckpt_fmt)
        if man_fmt.major != cand_fmt.major:
            reasons.append(
                f"checkpoint_format_version major mismatch: manifest "
                f"{manifest.checkpoint_format_version!r}, checkpoint "
                f"{ckpt_fmt!r}"
            )
            severity = _escalate(severity, SEVERITY_MAJOR)
        elif (man_fmt.minor, man_fmt.patch) != (cand_fmt.minor, cand_fmt.patch):
            reasons.append(
                f"checkpoint_format_version minor/patch drift: manifest "
                f"{manifest.checkpoint_format_version!r}, checkpoint "
                f"{ckpt_fmt!r}"
            )
            severity = _escalate(severity, SEVERITY_MINOR)

    cfg_ver = checkpoint_meta.get("config_version")
    if cfg_ver is not None:
        man_cfg = parse_semver(manifest.config_version)
        cand_cfg = parse_semver(cfg_ver)
        if man_cfg.major != cand_cfg.major:
            reasons.append(
                f"config_version major mismatch: manifest "
                f"{manifest.config_version!r}, checkpoint {cfg_ver!r}"
            )
            severity = _escalate(severity, SEVERITY_MAJOR)
        elif (man_cfg.minor, man_cfg.patch) != (cand_cfg.minor, cand_cfg.patch):
            reasons.append(
                f"config_version minor/patch drift: manifest "
                f"{manifest.config_version!r}, checkpoint {cfg_ver!r}"
            )
            severity = _escalate(severity, SEVERITY_MINOR)

    man_hash = manifest.tokenizer_hash
    ckpt_hash = checkpoint_meta.get("tokenizer_hash")
    if man_hash is not None and ckpt_hash is not None and man_hash != ckpt_hash:
        reasons.append(
            f"tokenizer_hash mismatch: manifest {man_hash!r}, checkpoint "
            f"{ckpt_hash!r}"
        )
        severity = _escalate(severity, SEVERITY_MAJOR)

    compatible = severity != SEVERITY_MAJOR
    return CompatibilityVerdict(
        compatible=compatible,
        reasons=tuple(reasons),
        severity=severity,
    )


def assert_compatible(verdict: Any) -> None:
    """Raise :class:`CompatibilityError` on a ``major_break`` verdict.

    Accepts a :class:`CompatibilityVerdict` and raises only when severity
    indicates a hard break. Minor mismatches are allowed to pass.
    """
    if not isinstance(verdict, CompatibilityVerdict):
        raise CompatibilityError(
            f"assert_compatible expected CompatibilityVerdict, got "
            f"{type(verdict).__name__}"
        )
    if verdict.severity == SEVERITY_MAJOR or not verdict.compatible:
        raise CompatibilityError(
            "incompatible: " + "; ".join(verdict.reasons)
            if verdict.reasons
            else "incompatible"
        )
