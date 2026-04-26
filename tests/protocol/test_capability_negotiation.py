"""Tests for src/protocol/capability_negotiation.py"""

from src.protocol.capability_negotiation import (
    PROTOCOL_REGISTRY,
    Capability,
    CapabilityNegotiator,
    CapabilitySet,
)

# ---------------------------------------------------------------------------
# CapabilitySet construction
# ---------------------------------------------------------------------------


def test_capability_set_stores_frozenset():
    cs = CapabilitySet(capabilities=frozenset({Capability.STREAMING}))
    assert isinstance(cs.capabilities, frozenset)


def test_capability_set_from_list_becomes_frozenset():
    cs = CapabilitySet(capabilities=frozenset([Capability.VISION, Capability.AUDIO]))
    assert Capability.VISION in cs.capabilities


def test_capability_set_default_version():
    cs = CapabilitySet(capabilities=frozenset())
    assert cs.version == "1.0"


# ---------------------------------------------------------------------------
# negotiate
# ---------------------------------------------------------------------------


def test_negotiate_intersection():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset({Capability.STREAMING, Capability.TOOL_USE}))
    server = CapabilitySet(capabilities=frozenset({Capability.STREAMING, Capability.VISION}))
    result = neg.negotiate(client, server)
    assert result.capabilities == frozenset({Capability.STREAMING})


def test_negotiate_empty_when_no_overlap():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset({Capability.AUDIO}))
    server = CapabilitySet(capabilities=frozenset({Capability.VISION}))
    result = neg.negotiate(client, server)
    assert len(result.capabilities) == 0


def test_negotiate_uses_server_version():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset({Capability.STREAMING}), version="0.9")
    server = CapabilitySet(capabilities=frozenset({Capability.STREAMING}), version="2.0")
    result = neg.negotiate(client, server)
    assert result.version == "2.0"


# ---------------------------------------------------------------------------
# is_compatible
# ---------------------------------------------------------------------------


def test_is_compatible_true_when_all_required_present():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset({Capability.STREAMING, Capability.TOOL_USE}))
    server = CapabilitySet(
        capabilities=frozenset({Capability.STREAMING, Capability.TOOL_USE, Capability.VISION})
    )
    assert neg.is_compatible(client, server, {Capability.STREAMING})


def test_is_compatible_false_when_required_missing():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset({Capability.STREAMING}))
    server = CapabilitySet(capabilities=frozenset({Capability.STREAMING}))
    assert not neg.is_compatible(client, server, {Capability.VISION})


def test_is_compatible_empty_required_always_true():
    neg = CapabilityNegotiator()
    client = CapabilitySet(capabilities=frozenset())
    server = CapabilitySet(capabilities=frozenset())
    assert neg.is_compatible(client, server, set())


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_structure():
    neg = CapabilityNegotiator()
    cs = CapabilitySet(capabilities=frozenset({Capability.CODE_EXEC}), version="1.1")
    d = neg.to_dict(cs)
    assert "capabilities" in d
    assert "version" in d
    assert d["version"] == "1.1"
    assert "code_exec" in d["capabilities"]


def test_from_dict_roundtrip():
    neg = CapabilityNegotiator()
    original = CapabilitySet(
        capabilities=frozenset({Capability.STREAMING, Capability.LONG_CONTEXT}),
        version="1.0",
    )
    restored = neg.from_dict(neg.to_dict(original))
    assert restored.capabilities == original.capabilities
    assert restored.version == original.version


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_protocol_registry_contains_capability_negotiator():
    assert "capability_negotiator" in PROTOCOL_REGISTRY
    assert isinstance(PROTOCOL_REGISTRY["capability_negotiator"], CapabilityNegotiator)


def test_all_six_capabilities_defined():
    members = {c.value for c in Capability}
    expected = {"streaming", "tool_use", "vision", "audio", "code_exec", "long_context"}
    assert members == expected
