"""Release-track routing for Aurelius family variants.

Under Meta-Prompt v5 (Model Family Edition), a :class:`ReleaseTrackRouter`
gates which family variants may be served under a given policy. Each
:class:`FamilyManifest` carries a ``release_track`` string (research /
beta / stable / deprecated); the router applies a :class:`RouterPolicy`
to that track and returns a :class:`RouteDecision` indicating whether the
variant may be served, plus any advisory warnings.

Policies shipped in this module:

    * ``PRODUCTION_POLICY`` -- allow STABLE only; warn on DEPRECATED;
      deny RESEARCH and BETA outright.
    * ``INTERNAL_POLICY``   -- allow STABLE and BETA; warn on DEPRECATED;
      require explicit override (``allow_research``) for RESEARCH.
    * ``DEV_POLICY``        -- allow every track; warn on DEPRECATED.

Siblings (not imported; referenced for context only):

    * ``src/model/variant_adapter.py``
    * ``src/model/checkpoint_migration.py``
    * ``src/agent/web_browse_tool.py``
    * ``src/data/tokenizer_contract.py``
    * ``src/serving/function_calling_api.py``

Pure stdlib: ``dataclasses``, ``typing``. No foreign imports; no new config
flags. Additive within its own file only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .family import MODEL_VARIANT_REGISTRY
from .manifest import FamilyManifest, ReleaseTrack

__all__ = [
    "RouterPolicy",
    "RouteDecision",
    "RouterOverrideError",
    "ReleaseTrackRouter",
    "PRODUCTION_POLICY",
    "INTERNAL_POLICY",
    "DEV_POLICY",
    "POLICY_REGISTRY",
]


class RouterOverrideError(Exception):
    """Raised when a required override flag is missing for a release track."""


@dataclass(frozen=True)
class RouterPolicy:
    """Policy specifying how each release track is handled.

    A track should appear in at most one of ``allowed_tracks``,
    ``warn_tracks``, ``require_override_tracks``, or ``deny_tracks``. A
    track that is absent from every set is treated as implicitly denied.

    ``warn_tracks`` grants access but attaches an advisory warning to the
    resulting :class:`RouteDecision`. ``require_override_tracks`` grants
    access only if the caller supplies a matching override flag (named
    ``allow_<track_value>``, e.g. ``allow_research``).
    """

    name: str
    allowed_tracks: tuple[ReleaseTrack, ...]
    warn_tracks: tuple[ReleaseTrack, ...]
    require_override_tracks: tuple[ReleaseTrack, ...]
    deny_tracks: tuple[ReleaseTrack, ...]


@dataclass
class RouteDecision:
    """Outcome of routing a single variant under a policy."""

    allowed: bool
    reason: str
    variant_id: str
    track: ReleaseTrack
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _override_flag_for(track: ReleaseTrack) -> str:
    return f"allow_{track.value}"


class ReleaseTrackRouter:
    """Apply a :class:`RouterPolicy` to a manifest's release track."""

    def __init__(self, policy: RouterPolicy) -> None:
        if not isinstance(policy, RouterPolicy):
            raise TypeError(f"policy must be RouterPolicy, got {type(policy).__name__}")
        self.policy = policy

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------
    def route(
        self,
        manifest: FamilyManifest,
        override_flags: set[str] | None = None,
    ) -> RouteDecision:
        """Return a :class:`RouteDecision` for ``manifest`` under policy."""
        if not isinstance(manifest, FamilyManifest):
            raise TypeError(f"manifest must be FamilyManifest, got {type(manifest).__name__}")
        flags = self._normalize_flags(override_flags)
        try:
            track = ReleaseTrack(manifest.release_track)
        except ValueError as exc:  # pragma: no cover - manifest validates
            raise ValueError(
                f"manifest.release_track {manifest.release_track!r} is not "
                f"a valid ReleaseTrack value"
            ) from exc

        variant_id = f"{manifest.family_name}/{manifest.variant_name}"
        policy = self.policy
        warnings: list[str] = []

        if track in policy.deny_tracks:
            return RouteDecision(
                allowed=False,
                reason=(f"policy {policy.name!r} denies release_track {track.value!r}"),
                variant_id=variant_id,
                track=track,
                warnings=tuple(warnings),
            )

        if track in policy.require_override_tracks:
            required = _override_flag_for(track)
            if required not in flags:
                return RouteDecision(
                    allowed=False,
                    reason=(
                        f"policy {policy.name!r} requires override flag "
                        f"{required!r} for release_track {track.value!r}"
                    ),
                    variant_id=variant_id,
                    track=track,
                    warnings=tuple(warnings),
                )
            warnings.append(f"override {required!r} granted for release_track {track.value!r}")
            return RouteDecision(
                allowed=True,
                reason=(
                    f"policy {policy.name!r} accepted release_track "
                    f"{track.value!r} via override {required!r}"
                ),
                variant_id=variant_id,
                track=track,
                warnings=tuple(warnings),
            )

        if track in policy.warn_tracks:
            warnings.append(
                f"release_track {track.value!r} is advisory under policy {policy.name!r}"
            )
            return RouteDecision(
                allowed=True,
                reason=(
                    f"policy {policy.name!r} allows release_track {track.value!r} with warning"
                ),
                variant_id=variant_id,
                track=track,
                warnings=tuple(warnings),
            )

        if track in policy.allowed_tracks:
            return RouteDecision(
                allowed=True,
                reason=(f"policy {policy.name!r} allows release_track {track.value!r}"),
                variant_id=variant_id,
                track=track,
                warnings=tuple(warnings),
            )

        # Track appears in no bucket: implicit deny.
        return RouteDecision(
            allowed=False,
            reason=(
                f"policy {policy.name!r} has no rule for release_track {track.value!r}; denying"
            ),
            variant_id=variant_id,
            track=track,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Variant-id convenience wrapper
    # ------------------------------------------------------------------
    def resolve_by_variant_id(
        self,
        variant_id: str,
        override_flags: set[str] | None = None,
    ) -> RouteDecision:
        """Look up ``variant_id`` in the registry and route its manifest."""
        if variant_id not in MODEL_VARIANT_REGISTRY:
            raise KeyError(
                f"variant id {variant_id!r} not registered; "
                f"known variants: {sorted(MODEL_VARIANT_REGISTRY)}"
            )
        variant = MODEL_VARIANT_REGISTRY[variant_id]
        return self.route(variant.manifest, override_flags=override_flags)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_flags(
        override_flags: Iterable[str] | None,
    ) -> frozenset[str]:
        if override_flags is None:
            return frozenset()
        if isinstance(override_flags, str):
            raise TypeError("override_flags must be a set of strings, got bare str")
        flags = frozenset(override_flags)
        for f in flags:
            if not isinstance(f, str):
                raise TypeError(f"override_flags entries must be str, got {type(f).__name__}")
        return flags


# ---------------------------------------------------------------------------
# Shipped policies
# ---------------------------------------------------------------------------

PRODUCTION_POLICY: RouterPolicy = RouterPolicy(
    name="production",
    allowed_tracks=(ReleaseTrack.STABLE,),
    warn_tracks=(ReleaseTrack.DEPRECATED,),
    require_override_tracks=(),
    deny_tracks=(ReleaseTrack.RESEARCH, ReleaseTrack.BETA),
)


INTERNAL_POLICY: RouterPolicy = RouterPolicy(
    name="internal",
    allowed_tracks=(ReleaseTrack.STABLE, ReleaseTrack.BETA),
    warn_tracks=(ReleaseTrack.DEPRECATED,),
    require_override_tracks=(ReleaseTrack.RESEARCH,),
    deny_tracks=(),
)


DEV_POLICY: RouterPolicy = RouterPolicy(
    name="dev",
    allowed_tracks=(
        ReleaseTrack.STABLE,
        ReleaseTrack.BETA,
        ReleaseTrack.RESEARCH,
    ),
    warn_tracks=(ReleaseTrack.DEPRECATED,),
    require_override_tracks=(),
    deny_tracks=(),
)


POLICY_REGISTRY: dict[str, RouterPolicy] = {
    PRODUCTION_POLICY.name: PRODUCTION_POLICY,
    INTERNAL_POLICY.name: INTERNAL_POLICY,
    DEV_POLICY.name: DEV_POLICY,
}
