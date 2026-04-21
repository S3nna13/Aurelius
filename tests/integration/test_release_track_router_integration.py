"""Integration tests: release-track router + model package surface."""

from __future__ import annotations

import src.model as model_pkg
from src.model import (
    AURELIUS_REFERENCE_MANIFEST,
    DEV_POLICY,
    INTERNAL_POLICY,
    POLICY_REGISTRY,
    PRODUCTION_POLICY,
    ReleaseTrackRouter,
)


def test_router_symbols_exported_from_model_package() -> None:
    for name in (
        "ReleaseTrackRouter",
        "RouterPolicy",
        "RouteDecision",
        "RouterOverrideError",
        "PRODUCTION_POLICY",
        "INTERNAL_POLICY",
        "DEV_POLICY",
        "POLICY_REGISTRY",
    ):
        assert hasattr(model_pkg, name), f"missing export: {name}"
        assert name in model_pkg.__all__, f"missing __all__ entry: {name}"


def test_policy_registry_three_entries_via_package() -> None:
    assert set(POLICY_REGISTRY) == {"production", "internal", "dev"}


def test_reference_manifest_routes_under_each_policy() -> None:
    # The reference manifest ships on the "research" track. Verify each
    # shipped policy routes it correctly.
    assert AURELIUS_REFERENCE_MANIFEST.release_track == "research"

    prod = ReleaseTrackRouter(PRODUCTION_POLICY).route(
        AURELIUS_REFERENCE_MANIFEST
    )
    assert prod.allowed is False

    internal_blocked = ReleaseTrackRouter(INTERNAL_POLICY).route(
        AURELIUS_REFERENCE_MANIFEST
    )
    assert internal_blocked.allowed is False
    internal_ok = ReleaseTrackRouter(INTERNAL_POLICY).route(
        AURELIUS_REFERENCE_MANIFEST, override_flags={"allow_research"}
    )
    assert internal_ok.allowed is True

    dev = ReleaseTrackRouter(DEV_POLICY).route(AURELIUS_REFERENCE_MANIFEST)
    assert dev.allowed is True


def test_resolve_by_variant_id_through_package_registry() -> None:
    router = ReleaseTrackRouter(DEV_POLICY)
    dec = router.resolve_by_variant_id(
        AURELIUS_REFERENCE_MANIFEST.registry_key
    )
    assert dec.allowed is True
    assert dec.variant_id == AURELIUS_REFERENCE_MANIFEST.registry_key
