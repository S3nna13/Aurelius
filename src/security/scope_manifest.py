"""Scope and policy enforcement for agent tool invocations.

Implements a deny-by-default pattern: unless a host, URL, or tool is
explicitly allowed by the ScopeManifest, all access is denied.
"""

import functools
import ipaddress
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum

try:
    from src.security import SECURITY_REGISTRY
except ImportError:
    SECURITY_REGISTRY: dict = {}


class ScanMode(StrEnum):
    PASSIVE_SAFE = "passive_safe"
    ACTIVE_APPROVED = "active_approved"


@dataclass
class RateLimits:
    requests_per_second: float = 1.0
    max_concurrent_tools: int = 3


@dataclass
class ScopeManifest:
    name: str
    domains: list  # allowed domain suffixes e.g. ["example.com"]
    hosts: list  # exact host allowlist
    cidrs: list  # CIDR ranges e.g. ["192.168.0.0/24"]
    out_of_scope_patterns: list  # regex patterns that disallow
    allow_tools: list  # explicit tool allowlist
    kill_switch_file: str = ".kill_switch"
    scan_mode: ScanMode = ScanMode.PASSIVE_SAFE
    rate_limits: RateLimits = field(default_factory=RateLimits)
    max_engagement_runtime_seconds: int = 3600


@functools.lru_cache(maxsize=128)
def _compile_patterns(patterns: tuple) -> tuple:
    """Cache compiled regex patterns keyed by the frozen tuple of pattern strings."""
    return tuple(re.compile(p) for p in patterns)


class ScopeChecker:
    """Stateless checker that evaluates hosts, URLs, and tools against a ScopeManifest."""

    def host_in_scope(self, manifest: ScopeManifest, host: str) -> bool:
        """Return True if *host* is explicitly allowed by the manifest.

        Handles:
        - IPv6 bracket notation  e.g. ``[::1]:8080``
        - Port suffixes          e.g. ``example.com:443``
        - Exact host matches
        - Domain suffix matches  (``sub.example.com`` matches ``example.com``)
        - CIDR membership
        """
        # Strip port and IPv6 brackets
        host = host.split(":")[0].strip("[]")

        # Exact match
        if host in manifest.hosts:
            return True

        # Domain suffix match
        for domain in manifest.domains:
            if host == domain or host.endswith("." + domain):
                return True

        # CIDR match (only meaningful for IP literals)
        try:
            addr = ipaddress.ip_address(host)
            for cidr in manifest.cidrs:
                if addr in ipaddress.ip_network(cidr, strict=False):
                    return True
        except ValueError:
            pass

        return False

    def url_in_scope(self, manifest: ScopeManifest, url: str) -> bool:
        """Return True if *url* is in scope per the manifest.

        Requires an allowed scheme, an in-scope host, and no match against
        any ``out_of_scope_patterns``.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            return False

        host = parsed.hostname or ""
        if not self.host_in_scope(manifest, host):
            return False

        # Check out_of_scope patterns against the full URL
        patterns = _compile_patterns(tuple(manifest.out_of_scope_patterns))
        for pat in patterns:
            if pat.search(url):
                return False

        return True

    def kill_switch_active(self, kill_switch_file: str) -> bool:
        """Return True if the kill-switch sentinel file exists on disk."""
        return os.path.exists(kill_switch_file)

    def allow_tool(self, manifest: ScopeManifest, tool_name: str) -> bool:
        """Return True only if *tool_name* appears in the explicit allowlist."""
        return tool_name in manifest.allow_tools


SECURITY_REGISTRY["scope_checker"] = ScopeChecker()

__all__ = [
    "ScanMode",
    "RateLimits",
    "ScopeManifest",
    "ScopeChecker",
    "SECURITY_REGISTRY",
]
