"""Tests for src.mcp.capability_catalog.

Coverage (≥ 28 tests):
- advertise() and get() round-trip
- get() returns None for absent capability
- list_capabilities() returns sorted names
- negotiate() all requested accepted
- negotiate() some rejected
- negotiate() optional_missing via negotiate_with_optional_tracking
- compatible_with() returns overlap
- compatible_with() returns empty for no overlap
- CapabilityVersion __str__
- CapabilityVersion __lt__ ordering
- CapabilityVersion equal versions are not less-than each other
- Overwrite same-name capability via advertise()
- Capability metadata field
- Capability optional field
- CAPABILITY_CATALOG_REGISTRY contains "default"
- CAPABILITY_CATALOG_REGISTRY["default"] is CapabilityCatalog
- negotiate empty requested list
- negotiate all absent
- compatible_with empty catalogs
- compatible_with self is fully compatible
- Multiple advertise/get cycles
"""

from __future__ import annotations

from src.mcp.capability_catalog import (
    CAPABILITY_CATALOG_REGISTRY,
    Capability,
    CapabilityCatalog,
    CapabilityVersion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(*caps: Capability) -> CapabilityCatalog:
    cat = CapabilityCatalog()
    for cap in caps:
        cat.advertise(cap)
    return cat


def _cap(name: str, major: int = 1, minor: int = 0, optional: bool = False) -> Capability:
    return Capability(name=name, version=CapabilityVersion(major, minor), optional=optional)


# ---------------------------------------------------------------------------
# CapabilityVersion
# ---------------------------------------------------------------------------


def test_capability_version_str():
    v = CapabilityVersion(1, 2, 3)
    assert str(v) == "1.2.3"


def test_capability_version_str_default_patch():
    v = CapabilityVersion(2, 0)
    assert str(v) == "2.0.0"


def test_capability_version_lt_true():
    assert CapabilityVersion(1, 0) < CapabilityVersion(2, 0)


def test_capability_version_lt_minor():
    assert CapabilityVersion(1, 0) < CapabilityVersion(1, 1)


def test_capability_version_lt_patch():
    assert CapabilityVersion(1, 0, 0) < CapabilityVersion(1, 0, 1)


def test_capability_version_lt_false_when_equal():
    v = CapabilityVersion(1, 2, 3)
    assert not (v < v)


def test_capability_version_lt_false_when_greater():
    assert not (CapabilityVersion(2, 0) < CapabilityVersion(1, 0))


def test_capability_version_ordering_sort():
    versions = [
        CapabilityVersion(2, 0),
        CapabilityVersion(1, 5),
        CapabilityVersion(1, 0),
    ]
    assert sorted(versions) == [
        CapabilityVersion(1, 0),
        CapabilityVersion(1, 5),
        CapabilityVersion(2, 0),
    ]


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------


def test_capability_optional_default_false():
    cap = _cap("streaming")
    assert cap.optional is False


def test_capability_optional_flag():
    cap = _cap("batching", optional=True)
    assert cap.optional is True


def test_capability_metadata_default_empty():
    cap = _cap("streaming")
    assert cap.metadata == {}


def test_capability_metadata_stored():
    cap = Capability(
        name="ext",
        version=CapabilityVersion(1, 0),
        metadata={"auth": "bearer"},
    )
    assert cap.metadata["auth"] == "bearer"


# ---------------------------------------------------------------------------
# CapabilityCatalog.advertise / get
# ---------------------------------------------------------------------------


def test_advertise_and_get_round_trip():
    cat = CapabilityCatalog()
    cap = _cap("streaming")
    cat.advertise(cap)
    assert cat.get("streaming") is cap


def test_get_absent_returns_none():
    cat = CapabilityCatalog()
    assert cat.get("ghost") is None


def test_advertise_overwrites_same_name():
    cat = CapabilityCatalog()
    old = _cap("streaming", major=1)
    new = _cap("streaming", major=2)
    cat.advertise(old)
    cat.advertise(new)
    assert cat.get("streaming") is new


def test_multiple_advertise_get_cycles():
    cat = CapabilityCatalog()
    for i in range(5):
        cat.advertise(_cap(f"cap{i}"))
    for i in range(5):
        assert cat.get(f"cap{i}") is not None


# ---------------------------------------------------------------------------
# list_capabilities
# ---------------------------------------------------------------------------


def test_list_capabilities_empty():
    cat = CapabilityCatalog()
    assert cat.list_capabilities() == []


def test_list_capabilities_sorted():
    cat = _make_catalog(_cap("zebra"), _cap("alpha"), _cap("mango"))
    assert cat.list_capabilities() == ["alpha", "mango", "zebra"]


def test_list_capabilities_single():
    cat = _make_catalog(_cap("only"))
    assert cat.list_capabilities() == ["only"]


# ---------------------------------------------------------------------------
# negotiate
# ---------------------------------------------------------------------------


def test_negotiate_all_accepted():
    cat = _make_catalog(_cap("streaming"), _cap("batching"))
    result = cat.negotiate(["streaming", "batching"])
    assert set(result["accepted"]) == {"streaming", "batching"}
    assert result["rejected"] == []
    assert result["optional_missing"] == []


def test_negotiate_some_rejected():
    cat = _make_catalog(_cap("streaming"))
    result = cat.negotiate(["streaming", "batching"])
    assert "streaming" in result["accepted"]
    assert "batching" in result["rejected"]


def test_negotiate_all_rejected():
    cat = CapabilityCatalog()
    result = cat.negotiate(["a", "b", "c"])
    assert result["accepted"] == []
    assert set(result["rejected"]) == {"a", "b", "c"}


def test_negotiate_empty_requested():
    cat = _make_catalog(_cap("streaming"))
    result = cat.negotiate([])
    assert result == {"accepted": [], "rejected": [], "optional_missing": []}


def test_negotiate_with_optional_tracking_optional_missing():
    cat = _make_catalog(_cap("streaming"))
    # "batching" is absent but known-optional via the side set
    result = cat.negotiate_with_optional_tracking(
        requested=["streaming", "batching"],
        optional_names={"batching"},
    )
    assert "streaming" in result["accepted"]
    assert result["rejected"] == []
    assert "batching" in result["optional_missing"]


def test_negotiate_with_optional_tracking_hard_rejected():
    cat = _make_catalog(_cap("streaming"))
    result = cat.negotiate_with_optional_tracking(
        requested=["streaming", "missing_required"],
        optional_names=set(),
    )
    assert "streaming" in result["accepted"]
    assert "missing_required" in result["rejected"]
    assert result["optional_missing"] == []


def test_negotiate_accepted_and_rejected_no_overlap():
    cat = _make_catalog(_cap("a"), _cap("b"))
    result = cat.negotiate(["a", "b", "c", "d"])
    assert set(result["accepted"]).isdisjoint(set(result["rejected"]))


# ---------------------------------------------------------------------------
# compatible_with
# ---------------------------------------------------------------------------


def test_compatible_with_returns_shared_names():
    cat1 = _make_catalog(_cap("streaming"), _cap("batching"))
    cat2 = _make_catalog(_cap("streaming"), _cap("tools"))
    shared = cat1.compatible_with(cat2)
    assert shared == ["streaming"]


def test_compatible_with_no_overlap_returns_empty():
    cat1 = _make_catalog(_cap("a"))
    cat2 = _make_catalog(_cap("b"))
    assert cat1.compatible_with(cat2) == []


def test_compatible_with_self_is_fully_compatible():
    cat = _make_catalog(_cap("x"), _cap("y"), _cap("z"))
    shared = cat.compatible_with(cat)
    assert shared == ["x", "y", "z"]


def test_compatible_with_empty_catalogs():
    cat1 = CapabilityCatalog()
    cat2 = CapabilityCatalog()
    assert cat1.compatible_with(cat2) == []


def test_compatible_with_returns_sorted():
    cat1 = _make_catalog(_cap("z"), _cap("a"), _cap("m"))
    cat2 = _make_catalog(_cap("z"), _cap("a"), _cap("m"))
    shared = cat1.compatible_with(cat2)
    assert shared == sorted(shared)


# ---------------------------------------------------------------------------
# Registry constant
# ---------------------------------------------------------------------------


def test_capability_catalog_registry_contains_default():
    assert "default" in CAPABILITY_CATALOG_REGISTRY


def test_capability_catalog_registry_default_is_class():
    assert CAPABILITY_CATALOG_REGISTRY["default"] is CapabilityCatalog


def test_capability_catalog_registry_is_dict():
    assert isinstance(CAPABILITY_CATALOG_REGISTRY, dict)
