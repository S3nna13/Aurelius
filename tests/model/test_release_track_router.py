"""Tests for :mod:`src.model.release_track_router`."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.model.manifest import (
    AURELIUS_REFERENCE_MANIFEST,
    FamilyManifest,
    ReleaseTrack,
)
from src.model.release_track_router import (
    DEV_POLICY,
    INTERNAL_POLICY,
    POLICY_REGISTRY,
    PRODUCTION_POLICY,
    ReleaseTrackRouter,
    RouteDecision,
    RouterPolicy,
)


# ---------------------------------------------------------------------------
# Manifest fixtures per track
# ---------------------------------------------------------------------------


def _mk_manifest(track: str, variant: str) -> FamilyManifest:
    return FamilyManifest(
        family_name="aurelius",
        variant_name=variant,
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
        release_track=track,
    )


STABLE_M = _mk_manifest("stable", "stable-x")
RESEARCH_M = _mk_manifest("research", "research-x")
BETA_M = _mk_manifest("beta", "beta-x")
DEPRECATED_M = _mk_manifest("deprecated", "deprecated-x")


# ---------------------------------------------------------------------------
# PRODUCTION_POLICY
# ---------------------------------------------------------------------------


def test_production_accepts_stable() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(STABLE_M)
    assert dec.allowed is True
    assert dec.track is ReleaseTrack.STABLE
    assert dec.variant_id == "aurelius/stable-x"


def test_production_rejects_research_without_override() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(RESEARCH_M)
    assert dec.allowed is False
    assert "research" in dec.reason


def test_production_rejects_research_even_with_override() -> None:
    # Production policy deny-lists research; override cannot rescue it.
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(RESEARCH_M, override_flags={"allow_research"})
    assert dec.allowed is False


def test_production_rejects_beta() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(BETA_M)
    assert dec.allowed is False


def test_production_warns_on_deprecated() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(DEPRECATED_M)
    assert dec.allowed is True
    assert len(dec.warnings) >= 1


# ---------------------------------------------------------------------------
# INTERNAL_POLICY
# ---------------------------------------------------------------------------


def test_internal_requires_override_for_research() -> None:
    router = ReleaseTrackRouter(INTERNAL_POLICY)
    blocked = router.route(RESEARCH_M)
    assert blocked.allowed is False
    assert "allow_research" in blocked.reason

    allowed = router.route(RESEARCH_M, override_flags={"allow_research"})
    assert allowed.allowed is True
    assert allowed.warnings  # override should leave a breadcrumb warning


def test_internal_allows_beta() -> None:
    router = ReleaseTrackRouter(INTERNAL_POLICY)
    dec = router.route(BETA_M)
    assert dec.allowed is True


# ---------------------------------------------------------------------------
# DEV_POLICY
# ---------------------------------------------------------------------------


def test_dev_accepts_all_tracks() -> None:
    router = ReleaseTrackRouter(DEV_POLICY)
    for m in (STABLE_M, BETA_M, RESEARCH_M, DEPRECATED_M):
        assert router.route(m).allowed is True


# ---------------------------------------------------------------------------
# POLICY_REGISTRY
# ---------------------------------------------------------------------------


def test_policy_registry_has_three_entries() -> None:
    assert set(POLICY_REGISTRY) == {"production", "internal", "dev"}
    assert POLICY_REGISTRY["production"] is PRODUCTION_POLICY
    assert POLICY_REGISTRY["internal"] is INTERNAL_POLICY
    assert POLICY_REGISTRY["dev"] is DEV_POLICY


# ---------------------------------------------------------------------------
# resolve_by_variant_id
# ---------------------------------------------------------------------------


def test_resolve_by_variant_id_hits_reference_variant() -> None:
    router = ReleaseTrackRouter(DEV_POLICY)
    key = AURELIUS_REFERENCE_MANIFEST.registry_key
    dec = router.resolve_by_variant_id(key)
    assert dec.variant_id == key
    assert dec.allowed is True


def test_resolve_by_variant_id_unknown_raises() -> None:
    router = ReleaseTrackRouter(DEV_POLICY)
    with pytest.raises(KeyError):
        router.resolve_by_variant_id("nope/does-not-exist")


# ---------------------------------------------------------------------------
# RouteDecision shape
# ---------------------------------------------------------------------------


def test_route_decision_fields_populated() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(STABLE_M)
    assert isinstance(dec, RouteDecision)
    assert isinstance(dec.allowed, bool)
    assert isinstance(dec.reason, str) and dec.reason
    assert dec.variant_id == "aurelius/stable-x"
    assert dec.track is ReleaseTrack.STABLE
    assert isinstance(dec.warnings, tuple)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_policy_rejects_everything() -> None:
    empty = RouterPolicy(
        name="empty",
        allowed_tracks=(),
        warn_tracks=(),
        require_override_tracks=(),
        deny_tracks=(),
    )
    router = ReleaseTrackRouter(empty)
    for m in (STABLE_M, BETA_M, RESEARCH_M, DEPRECATED_M):
        assert router.route(m).allowed is False


def test_warnings_tuple_non_empty_when_expected() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    dec = router.route(DEPRECATED_M)
    assert dec.warnings
    assert all(isinstance(w, str) for w in dec.warnings)


def test_route_is_deterministic() -> None:
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    a = router.route(DEPRECATED_M)
    b = router.route(DEPRECATED_M)
    assert a == b


def test_custom_policy_composes() -> None:
    custom = RouterPolicy(
        name="custom",
        allowed_tracks=(ReleaseTrack.STABLE,),
        warn_tracks=(ReleaseTrack.BETA,),
        require_override_tracks=(ReleaseTrack.RESEARCH,),
        deny_tracks=(ReleaseTrack.DEPRECATED,),
    )
    router = ReleaseTrackRouter(custom)
    assert router.route(STABLE_M).allowed is True
    beta = router.route(BETA_M)
    assert beta.allowed is True and beta.warnings
    assert router.route(RESEARCH_M).allowed is False
    assert (
        router.route(RESEARCH_M, override_flags={"allow_research"}).allowed
        is True
    )
    assert router.route(DEPRECATED_M).allowed is False


def test_override_flags_case_sensitive() -> None:
    router = ReleaseTrackRouter(INTERNAL_POLICY)
    wrong = router.route(RESEARCH_M, override_flags={"ALLOW_RESEARCH"})
    assert wrong.allowed is False
    right = router.route(RESEARCH_M, override_flags={"allow_research"})
    assert right.allowed is True


def test_router_rejects_non_policy() -> None:
    with pytest.raises(TypeError):
        ReleaseTrackRouter("production")  # type: ignore[arg-type]


def test_route_rejects_non_manifest() -> None:
    router = ReleaseTrackRouter(DEV_POLICY)
    with pytest.raises(TypeError):
        router.route("not-a-manifest")  # type: ignore[arg-type]


def test_override_flags_rejects_bare_str() -> None:
    router = ReleaseTrackRouter(INTERNAL_POLICY)
    with pytest.raises(TypeError):
        router.route(RESEARCH_M, override_flags="allow_research")  # type: ignore[arg-type]


def test_replace_track_roundtrip() -> None:
    # Sanity check that dataclasses.replace on the manifest works and the
    # router re-routes accordingly.
    m2 = replace(STABLE_M, release_track="research")
    router = ReleaseTrackRouter(PRODUCTION_POLICY)
    assert router.route(m2).allowed is False
