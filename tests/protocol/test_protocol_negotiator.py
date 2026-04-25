"""Tests for src/protocol/protocol_negotiator.py"""

import pytest
from src.protocol.protocol_negotiator import (
    NegotiationOutcome,
    ProtocolCapability,
    ProtocolNegotiator,
    ProtocolVersion,
    PROTOCOL_NEGOTIATOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neg(versions=None, caps=None):
    versions = versions or [ProtocolVersion(1, 0), ProtocolVersion(2, 0)]
    caps     = caps     or []
    return ProtocolNegotiator(supported_versions=versions, capabilities=caps)


# ---------------------------------------------------------------------------
# ProtocolVersion – __str__
# ---------------------------------------------------------------------------


def test_protocol_version_str_default_patch():
    v = ProtocolVersion(1, 2)
    assert str(v) == "1.2.0"


def test_protocol_version_str_explicit_patch():
    v = ProtocolVersion(3, 4, 5)
    assert str(v) == "3.4.5"


def test_protocol_version_str_zero():
    v = ProtocolVersion(0, 0, 0)
    assert str(v) == "0.0.0"


# ---------------------------------------------------------------------------
# ProtocolVersion – comparisons
# ---------------------------------------------------------------------------


def test_protocol_version_le_equal():
    assert ProtocolVersion(1, 0) <= ProtocolVersion(1, 0)


def test_protocol_version_le_less():
    assert ProtocolVersion(1, 0) <= ProtocolVersion(2, 0)


def test_protocol_version_le_not_greater():
    assert not (ProtocolVersion(2, 0) <= ProtocolVersion(1, 0))


def test_protocol_version_ge_equal():
    assert ProtocolVersion(2, 3) >= ProtocolVersion(2, 3)


def test_protocol_version_ge_greater():
    assert ProtocolVersion(3, 0) >= ProtocolVersion(2, 9)


def test_protocol_version_ge_not_less():
    assert not (ProtocolVersion(1, 0) >= ProtocolVersion(1, 1))


def test_protocol_version_patch_ordering():
    assert ProtocolVersion(1, 0, 0) <= ProtocolVersion(1, 0, 1)
    assert ProtocolVersion(1, 0, 5) >= ProtocolVersion(1, 0, 4)


# ---------------------------------------------------------------------------
# highest_common()
# ---------------------------------------------------------------------------


def test_highest_common_exact_overlap():
    neg = _neg()
    a   = [ProtocolVersion(1, 0), ProtocolVersion(2, 0)]
    b   = [ProtocolVersion(1, 0), ProtocolVersion(2, 0)]
    assert neg.highest_common(a, b) == ProtocolVersion(2, 0)


def test_highest_common_partial_overlap():
    neg = _neg()
    a   = [ProtocolVersion(1, 0), ProtocolVersion(2, 0)]
    b   = [ProtocolVersion(1, 0)]
    assert neg.highest_common(a, b) == ProtocolVersion(1, 0)


def test_highest_common_no_overlap_returns_none():
    neg = _neg()
    a   = [ProtocolVersion(1, 0)]
    b   = [ProtocolVersion(2, 0)]
    assert neg.highest_common(a, b) is None


def test_highest_common_selects_max():
    neg    = _neg()
    shared = [ProtocolVersion(1, 0), ProtocolVersion(1, 1), ProtocolVersion(2, 0)]
    assert neg.highest_common(shared, shared) == ProtocolVersion(2, 0)


def test_highest_common_single_element():
    neg = _neg()
    v   = ProtocolVersion(4, 2, 1)
    assert neg.highest_common([v], [v]) == v


# ---------------------------------------------------------------------------
# negotiate() – ACCEPTED
# ---------------------------------------------------------------------------


def test_negotiate_exact_version_match_accepted():
    neg    = _neg(versions=[ProtocolVersion(1, 0)])
    result = neg.negotiate([ProtocolVersion(1, 0)], [])
    assert result["outcome"] == NegotiationOutcome.ACCEPTED.value


def test_negotiate_accepted_version_string_correct():
    neg    = _neg(versions=[ProtocolVersion(1, 2, 3)])
    result = neg.negotiate([ProtocolVersion(1, 2, 3)], [])
    assert result["version"] == "1.2.3"


def test_negotiate_accepted_no_missing():
    neg    = _neg(versions=[ProtocolVersion(1, 0)], caps=[])
    result = neg.negotiate([ProtocolVersion(1, 0)], [])
    assert result["missing_required"] == []
    assert result["optional_missing"] == []


# ---------------------------------------------------------------------------
# negotiate() – REJECTED (no common version)
# ---------------------------------------------------------------------------


def test_negotiate_no_common_version_rejected():
    neg    = _neg(versions=[ProtocolVersion(1, 0)])
    result = neg.negotiate([ProtocolVersion(2, 0)], [])
    assert result["outcome"] == NegotiationOutcome.REJECTED.value


def test_negotiate_no_common_version_none():
    neg    = _neg(versions=[ProtocolVersion(1, 0)])
    result = neg.negotiate([ProtocolVersion(2, 0)], [])
    assert result["version"] is None


# ---------------------------------------------------------------------------
# negotiate() – REJECTED (missing required capability)
# ---------------------------------------------------------------------------


def test_negotiate_missing_required_rejected():
    caps = [ProtocolCapability("streaming", required=True)]
    neg  = _neg(caps=caps)
    result = neg.negotiate(
        [ProtocolVersion(1, 0)],
        [],   # client offers nothing
    )
    assert result["outcome"] == NegotiationOutcome.REJECTED.value


def test_negotiate_missing_required_listed():
    caps = [
        ProtocolCapability("streaming", required=True),
        ProtocolCapability("vision",    required=True),
    ]
    neg    = _neg(caps=caps)
    result = neg.negotiate([ProtocolVersion(1, 0)], ["streaming"])
    assert "vision" in result["missing_required"]


def test_negotiate_missing_required_version_is_none():
    caps = [ProtocolCapability("tool_use", required=True)]
    neg  = _neg(caps=caps)
    result = neg.negotiate([ProtocolVersion(1, 0)], [])
    assert result["version"] is None


# ---------------------------------------------------------------------------
# negotiate() – optional missing → ACCEPTED
# ---------------------------------------------------------------------------


def test_negotiate_optional_missing_accepted():
    caps = [ProtocolCapability("streaming", required=False)]
    neg  = _neg(caps=caps)
    result = neg.negotiate([ProtocolVersion(1, 0)], [])
    assert result["outcome"] == NegotiationOutcome.ACCEPTED.value


def test_negotiate_optional_missing_listed():
    caps = [ProtocolCapability("vision", required=False)]
    neg  = _neg(caps=caps)
    result = neg.negotiate([ProtocolVersion(1, 0)], [])
    assert "vision" in result["optional_missing"]


def test_negotiate_optional_present_not_in_missing():
    caps = [ProtocolCapability("streaming", required=False)]
    neg  = _neg(caps=caps)
    result = neg.negotiate([ProtocolVersion(1, 0)], ["streaming"])
    assert result["optional_missing"] == []


# ---------------------------------------------------------------------------
# negotiate() – highest version selected
# ---------------------------------------------------------------------------


def test_negotiate_highest_common_version_selected():
    neg = _neg(versions=[ProtocolVersion(1, 0), ProtocolVersion(2, 0), ProtocolVersion(3, 0)])
    result = neg.negotiate(
        [ProtocolVersion(1, 0), ProtocolVersion(2, 0)],
        [],
    )
    assert result["version"] == "2.0.0"


# ---------------------------------------------------------------------------
# NegotiationOutcome enum values
# ---------------------------------------------------------------------------


def test_outcome_accepted_value():
    assert NegotiationOutcome.ACCEPTED.value == "accepted"


def test_outcome_rejected_value():
    assert NegotiationOutcome.REJECTED.value == "rejected"


def test_outcome_downgraded_value():
    assert NegotiationOutcome.DOWNGRADED.value == "downgraded"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in PROTOCOL_NEGOTIATOR_REGISTRY


def test_registry_default_is_class():
    assert PROTOCOL_NEGOTIATOR_REGISTRY["default"] is ProtocolNegotiator


def test_registry_default_instantiable():
    cls = PROTOCOL_NEGOTIATOR_REGISTRY["default"]
    neg = cls(supported_versions=[ProtocolVersion(1, 0)], capabilities=[])
    assert isinstance(neg, ProtocolNegotiator)
