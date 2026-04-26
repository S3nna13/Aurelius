"""Tests for protocol negotiator."""

from __future__ import annotations

from src.protocol.negotiator import ProtocolNegotiator, ProtocolVersion


class TestProtocolVersion:
    def test_compatible_same_major_and_minor(self):
        v1 = ProtocolVersion(1, 0)
        v2 = ProtocolVersion(1, 0)
        assert v1.compatible_with(v2) is True

    def test_incompatible_different_major(self):
        v1 = ProtocolVersion(1, 0)
        v2 = ProtocolVersion(2, 0)
        assert v1.compatible_with(v2) is False

    def test_parse(self):
        v = ProtocolVersion.parse("2.1.3")
        assert v.major == 2
        assert v.minor == 1
        assert v.patch == 3


class TestProtocolNegotiator:
    def test_negotiate_success(self):
        pn = ProtocolNegotiator()
        pn.register("api", ProtocolVersion(1, 0))
        ok, msg = pn.negotiate("api", ProtocolVersion(1, 0))
        assert ok is True

    def test_negotiate_unknown_service(self):
        pn = ProtocolNegotiator()
        ok, _ = pn.negotiate("unknown", ProtocolVersion(1, 0))
        assert ok is False

    def test_negotiate_incompatible(self):
        pn = ProtocolNegotiator()
        pn.register("api", ProtocolVersion(2, 0))
        ok, _ = pn.negotiate("api", ProtocolVersion(1, 0))
        assert ok is False
