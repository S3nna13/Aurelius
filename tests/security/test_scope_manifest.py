"""Tests for src/security/scope_manifest.py — 12+ cases."""

import os
import tempfile
import pytest

from src.security.scope_manifest import (
    RateLimits,
    ScanMode,
    ScopeChecker,
    ScopeManifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def checker() -> ScopeChecker:
    return ScopeChecker()


@pytest.fixture()
def manifest() -> ScopeManifest:
    return ScopeManifest(
        name="test-manifest",
        domains=["example.com", "internal.corp"],
        hosts=["exact-host.io", "192.168.1.50"],
        cidrs=["10.0.0.0/8", "172.16.0.0/12"],
        out_of_scope_patterns=[r"/admin", r"logout"],
        allow_tools=["nmap", "burp", "ffuf"],
        kill_switch_file=".ks_test",
    )


# ---------------------------------------------------------------------------
# host_in_scope
# ---------------------------------------------------------------------------

class TestHostInScope:
    def test_exact_host_match(self, checker, manifest):
        assert checker.host_in_scope(manifest, "exact-host.io") is True

    def test_exact_ip_host_match(self, checker, manifest):
        assert checker.host_in_scope(manifest, "192.168.1.50") is True

    def test_domain_suffix_match(self, checker, manifest):
        assert checker.host_in_scope(manifest, "sub.example.com") is True

    def test_domain_exact_match(self, checker, manifest):
        assert checker.host_in_scope(manifest, "example.com") is True

    def test_subdomain_two_levels(self, checker, manifest):
        assert checker.host_in_scope(manifest, "a.b.internal.corp") is True

    def test_host_not_in_scope(self, checker, manifest):
        assert checker.host_in_scope(manifest, "evil.com") is False

    def test_partial_domain_not_matched(self, checker, manifest):
        # "notexample.com" must NOT match domain "example.com"
        assert checker.host_in_scope(manifest, "notexample.com") is False

    def test_cidr_match_10_block(self, checker, manifest):
        assert checker.host_in_scope(manifest, "10.42.100.1") is True

    def test_cidr_match_172_block(self, checker, manifest):
        assert checker.host_in_scope(manifest, "172.20.5.5") is True

    def test_cidr_no_match(self, checker, manifest):
        assert checker.host_in_scope(manifest, "8.8.8.8") is False

    def test_host_with_port_stripped(self, checker, manifest):
        # "exact-host.io:8080" → host stripped to "exact-host.io"
        assert checker.host_in_scope(manifest, "exact-host.io:8080") is True

    def test_ipv6_brackets_stripped(self, checker, manifest):
        # IPv6 host not in manifest → False; ensures bracket stripping runs without error
        result = checker.host_in_scope(manifest, "[::1]:9090")
        assert isinstance(result, bool)

    def test_empty_host_not_in_scope(self, checker, manifest):
        assert checker.host_in_scope(manifest, "") is False


# ---------------------------------------------------------------------------
# url_in_scope
# ---------------------------------------------------------------------------

class TestUrlInScope:
    def test_http_url_in_scope(self, checker, manifest):
        assert checker.url_in_scope(manifest, "http://sub.example.com/path") is True

    def test_https_url_in_scope(self, checker, manifest):
        assert checker.url_in_scope(manifest, "https://exact-host.io/api") is True

    def test_ws_url_in_scope(self, checker, manifest):
        assert checker.url_in_scope(manifest, "ws://sub.example.com/ws") is True

    def test_wss_url_in_scope(self, checker, manifest):
        assert checker.url_in_scope(manifest, "wss://sub.example.com/ws") is True

    def test_ftp_scheme_rejected(self, checker, manifest):
        assert checker.url_in_scope(manifest, "ftp://sub.example.com/file") is False

    def test_file_scheme_rejected(self, checker, manifest):
        assert checker.url_in_scope(manifest, "file:///etc/passwd") is False

    def test_out_of_scope_pattern_admin(self, checker, manifest):
        # "/admin" path triggers out_of_scope_patterns
        assert checker.url_in_scope(manifest, "https://sub.example.com/admin/panel") is False

    def test_out_of_scope_pattern_logout(self, checker, manifest):
        assert checker.url_in_scope(manifest, "https://exact-host.io/logout") is False

    def test_url_unknown_host(self, checker, manifest):
        assert checker.url_in_scope(manifest, "https://unknown.example.org/page") is False

    def test_cidr_ip_url_in_scope(self, checker, manifest):
        assert checker.url_in_scope(manifest, "http://10.1.2.3/endpoint") is True


# ---------------------------------------------------------------------------
# kill_switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_switch_inactive_when_file_absent(self, checker):
        assert checker.kill_switch_active("/tmp/__no_such_kill_switch_file__") is False

    def test_kill_switch_active_when_file_present(self, checker):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            assert checker.kill_switch_active(tmp_path) is True
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# allow_tool
# ---------------------------------------------------------------------------

class TestAllowTool:
    def test_allowed_tool(self, checker, manifest):
        assert checker.allow_tool(manifest, "nmap") is True

    def test_not_allowed_tool(self, checker, manifest):
        assert checker.allow_tool(manifest, "metasploit") is False

    def test_empty_tool_name(self, checker, manifest):
        assert checker.allow_tool(manifest, "") is False


# ---------------------------------------------------------------------------
# ScopeManifest defaults & dataclass integrity
# ---------------------------------------------------------------------------

class TestScopeManifestDefaults:
    def test_default_scan_mode(self):
        m = ScopeManifest(
            name="x", domains=[], hosts=[], cidrs=[],
            out_of_scope_patterns=[], allow_tools=[],
        )
        assert m.scan_mode == ScanMode.PASSIVE_SAFE

    def test_default_kill_switch_file(self):
        m = ScopeManifest(
            name="x", domains=[], hosts=[], cidrs=[],
            out_of_scope_patterns=[], allow_tools=[],
        )
        assert m.kill_switch_file == ".kill_switch"

    def test_rate_limits_defaults(self):
        rl = RateLimits()
        assert rl.requests_per_second == 1.0
        assert rl.max_concurrent_tools == 3

    def test_scan_mode_values(self):
        assert ScanMode.PASSIVE_SAFE == "passive_safe"
        assert ScanMode.ACTIVE_APPROVED == "active_approved"
