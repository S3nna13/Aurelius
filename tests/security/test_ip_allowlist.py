"""Tests for src/security/ip_allowlist.py — ≥28 tests."""

from __future__ import annotations

import pytest

from src.security.ip_allowlist import (
    CIDRBlock,
    IP_ALLOWLIST_REGISTRY,
    IPAllowlist,
)


# ---------------------------------------------------------------------------
# CIDRBlock — construction
# ---------------------------------------------------------------------------

class TestCIDRBlockConstruction:
    def test_from_string_slash24(self):
        block = CIDRBlock.from_string("192.168.1.0/24")
        assert block.network == "192.168.1.0"
        assert block.prefix_len == 24

    def test_from_string_slash16(self):
        block = CIDRBlock.from_string("10.0.0.0/16")
        assert block.network == "10.0.0.0"
        assert block.prefix_len == 16

    def test_from_string_slash32(self):
        block = CIDRBlock.from_string("1.2.3.4/32")
        assert block.network == "1.2.3.4"
        assert block.prefix_len == 32

    def test_from_string_slash0(self):
        block = CIDRBlock.from_string("0.0.0.0/0")
        assert block.prefix_len == 0

    def test_from_string_normalises_host_bits(self):
        # 192.168.1.5/24 should normalise to 192.168.1.0
        block = CIDRBlock.from_string("192.168.1.5/24")
        assert block.network == "192.168.1.0"

    def test_missing_slash_raises(self):
        with pytest.raises(ValueError):
            CIDRBlock.from_string("192.168.1.0")

    def test_invalid_prefix_len_raises(self):
        with pytest.raises(ValueError):
            CIDRBlock.from_string("10.0.0.0/33")

    def test_frozen(self):
        block = CIDRBlock.from_string("10.0.0.0/8")
        with pytest.raises((AttributeError, TypeError)):
            block.prefix_len = 16  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CIDRBlock.contains
# ---------------------------------------------------------------------------

class TestCIDRBlockContains:
    def test_contains_ip_in_slash24(self):
        block = CIDRBlock.from_string("192.168.1.0/24")
        assert block.contains("192.168.1.100") is True

    def test_contains_network_address_itself(self):
        block = CIDRBlock.from_string("192.168.1.0/24")
        assert block.contains("192.168.1.0") is True

    def test_contains_broadcast_address(self):
        block = CIDRBlock.from_string("192.168.1.0/24")
        assert block.contains("192.168.1.255") is True

    def test_not_contains_ip_outside_slash24(self):
        block = CIDRBlock.from_string("192.168.1.0/24")
        assert block.contains("192.168.2.1") is False

    def test_slash32_exact_match_true(self):
        block = CIDRBlock.from_string("203.0.113.42/32")
        assert block.contains("203.0.113.42") is True

    def test_slash32_exact_match_false(self):
        block = CIDRBlock.from_string("203.0.113.42/32")
        assert block.contains("203.0.113.43") is False

    def test_slash0_contains_all(self):
        block = CIDRBlock.from_string("0.0.0.0/0")
        assert block.contains("1.2.3.4") is True
        assert block.contains("255.255.255.255") is True
        assert block.contains("0.0.0.0") is True

    def test_loopback_in_range(self):
        block = CIDRBlock.from_string("127.0.0.0/8")
        assert block.contains("127.0.0.1") is True

    def test_loopback_not_in_private_range(self):
        block = CIDRBlock.from_string("192.168.0.0/16")
        assert block.contains("127.0.0.1") is False


# ---------------------------------------------------------------------------
# IPAllowlist — default behaviour
# ---------------------------------------------------------------------------

class TestIPAllowlistDefaults:
    def test_empty_allowlist_allows_all(self):
        al = IPAllowlist()
        assert al.is_allowed("8.8.8.8") is True

    def test_empty_allowlist_allows_loopback(self):
        al = IPAllowlist()
        assert al.is_allowed("127.0.0.1") is True


# ---------------------------------------------------------------------------
# IPAllowlist — allow blocks
# ---------------------------------------------------------------------------

class TestIPAllowlistAllow:
    def test_allow_block_permits_ip_in_range(self):
        al = IPAllowlist()
        al.allow("192.168.1.0/24")
        assert al.is_allowed("192.168.1.50") is True

    def test_allow_block_rejects_ip_outside_range(self):
        al = IPAllowlist()
        al.allow("192.168.1.0/24")
        assert al.is_allowed("10.0.0.1") is False

    def test_multiple_allow_blocks(self):
        al = IPAllowlist()
        al.allow("10.0.0.0/8")
        al.allow("192.168.0.0/16")
        assert al.is_allowed("10.5.5.5") is True
        assert al.is_allowed("192.168.99.1") is True
        assert al.is_allowed("8.8.8.8") is False


# ---------------------------------------------------------------------------
# IPAllowlist — deny blocks
# ---------------------------------------------------------------------------

class TestIPAllowlistDeny:
    def test_deny_block_blocks_ip(self):
        al = IPAllowlist()
        al.deny("10.0.0.0/8")
        assert al.is_allowed("10.1.2.3") is False

    def test_deny_wins_over_allow(self):
        al = IPAllowlist()
        al.allow("10.0.0.0/8")
        al.deny("10.0.0.0/8")
        assert al.is_allowed("10.5.5.5") is False

    def test_deny_specific_ip_slash32(self):
        al = IPAllowlist()
        al.allow("192.168.1.0/24")
        al.deny("192.168.1.99/32")
        assert al.is_allowed("192.168.1.99") is False
        assert al.is_allowed("192.168.1.100") is True


# ---------------------------------------------------------------------------
# IPAllowlist.check
# ---------------------------------------------------------------------------

class TestIPAllowlistCheck:
    def test_check_returns_dict_with_required_keys(self):
        al = IPAllowlist()
        result = al.check("8.8.8.8")
        assert "ip" in result
        assert "allowed" in result
        assert "matched_allow" in result
        assert "matched_deny" in result

    def test_check_ip_field(self):
        al = IPAllowlist()
        result = al.check("1.2.3.4")
        assert result["ip"] == "1.2.3.4"

    def test_check_matched_deny_populated(self):
        al = IPAllowlist()
        al.deny("10.0.0.0/8")
        result = al.check("10.5.5.5")
        assert result["matched_deny"] is not None
        assert result["allowed"] is False

    def test_check_matched_allow_populated(self):
        al = IPAllowlist()
        al.allow("192.168.0.0/16")
        result = al.check("192.168.1.1")
        assert result["matched_allow"] is not None
        assert result["allowed"] is True

    def test_check_no_match_empty_allowlist(self):
        al = IPAllowlist()
        result = al.check("5.5.5.5")
        assert result["allowed"] is True
        assert result["matched_deny"] is None


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in IP_ALLOWLIST_REGISTRY

    def test_registry_default_is_class(self):
        cls = IP_ALLOWLIST_REGISTRY["default"]
        al = cls()
        assert isinstance(al, IPAllowlist)
