"""Tests for capability_negotiator."""

from __future__ import annotations

import pytest

from src.mcp.capability_negotiator import CAPABILITY_NEGOTIATOR_REGISTRY, CapabilityNegotiator


class TestCapabilityNegotiator:
    def test_negotiate_intersection(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["tools.list", "tools.call", "resources.read"])
        assert n.negotiate(["tools.list", "tools.call", "prompts.get"]) == [
            "tools.call",
            "tools.list",
        ]

    def test_negotiate_empty_remote(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["tools.list"])
        assert n.negotiate([]) == []

    def test_can_handle_true(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["tools.list"])
        assert n.can_handle("tools.list")

    def test_can_handle_false(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["tools.list"])
        assert not n.can_handle("tools.call")

    def test_list_supported_sorted(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["b", "a", "c"])
        assert n.list_supported() == ["a", "b", "c"]

    def test_add_capability(self):
        n = CapabilityNegotiator()
        n.add_capability("tools.list")
        assert n.can_handle("tools.list")

    def test_remove_capability(self):
        n = CapabilityNegotiator()
        n.offer_capabilities(["tools.list", "tools.call"])
        n.remove_capability("tools.list")
        assert not n.can_handle("tools.list")
        assert n.can_handle("tools.call")

    def test_remove_missing_no_error(self):
        n = CapabilityNegotiator()
        n.remove_capability("missing")
        assert n.list_supported() == []


class TestCapabilityNegotiatorValidation:
    def test_empty_string_raises(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.add_capability("")

    def test_too_long_raises(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.add_capability("a" * 65)

    def test_invalid_chars_raises(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.add_capability("tools/list")

    def test_non_string_raises(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.add_capability(123)  # type: ignore[arg-type]

    def test_offer_capabilities_validates(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.offer_capabilities(["valid", ""])

    def test_negotiate_validates_remote(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.negotiate(["valid", ""])

    def test_can_handle_validates(self):
        n = CapabilityNegotiator()
        with pytest.raises(ValueError):
            n.can_handle("")


class TestCapabilityNegotiatorRegistry:
    def test_registry_is_dict(self):
        assert isinstance(CAPABILITY_NEGOTIATOR_REGISTRY, dict)
