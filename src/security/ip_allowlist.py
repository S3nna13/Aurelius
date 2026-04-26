"""IP allowlist / denylist with CIDR support.

Manual CIDR parsing using only stdlib (socket + struct).  No third-party
dependencies required.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CIDRBlock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIDRBlock:
    """Represents a single IPv4 CIDR block."""

    network: str  # dotted-decimal network address, e.g. "192.168.1.0"
    prefix_len: int  # 0-32

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, cidr: str) -> CIDRBlock:
        """Parse a CIDR string such as ``"192.168.1.0/24"``."""
        if "/" not in cidr:
            raise ValueError(f"Invalid CIDR string (missing '/'): {cidr!r}")
        network_str, prefix_str = cidr.split("/", 1)
        prefix_len = int(prefix_str)
        if not (0 <= prefix_len <= 32):
            raise ValueError(f"Prefix length out of range: {prefix_len}")
        # Normalise: mask off host bits
        ip_int = struct.unpack("!I", socket.inet_aton(network_str))[0]
        mask = _make_mask(prefix_len)
        network_int = ip_int & mask
        # Convert back to dotted notation
        network_normalised = socket.inet_ntoa(struct.pack("!I", network_int))
        return cls(network=network_normalised, prefix_len=prefix_len)

    # ------------------------------------------------------------------
    # Membership test
    # ------------------------------------------------------------------

    def contains(self, ip: str) -> bool:
        """Return *True* if *ip* falls within this CIDR block."""
        try:
            ip_int = struct.unpack("!I", socket.inet_aton(ip))[0]
        except OSError:
            return False
        mask = _make_mask(self.prefix_len)
        network_int = struct.unpack("!I", socket.inet_aton(self.network))[0]
        return (ip_int & mask) == network_int

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.network}/{self.prefix_len}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask(prefix_len: int) -> int:
    """Return a 32-bit mask for the given prefix length."""
    if prefix_len == 0:
        return 0
    return (~((1 << (32 - prefix_len)) - 1)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# IPAllowlist
# ---------------------------------------------------------------------------


class IPAllowlist:
    """Manage an IPv4 allowlist and denylist with CIDR range support.

    Logic:
    - If the IP matches any *deny* block → blocked (deny wins).
    - If the allowlist is empty, all non-denied IPs are allowed by default.
    - If the allowlist is non-empty, the IP must match at least one *allow*
      block to be permitted.
    """

    def __init__(self) -> None:
        self._allow: list[CIDRBlock] = []
        self._deny: list[CIDRBlock] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def allow(self, cidr: str) -> None:
        """Add *cidr* to the allowlist."""
        self._allow.append(CIDRBlock.from_string(cidr))

    def deny(self, cidr: str) -> None:
        """Add *cidr* to the denylist."""
        self._deny.append(CIDRBlock.from_string(cidr))

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def is_allowed(self, ip: str) -> bool:
        """Return *True* if *ip* is permitted."""
        # Deny wins unconditionally
        for block in self._deny:
            if block.contains(ip):
                return False
        # Default-allow when allowlist is empty
        if not self._allow:
            return True
        # Must match at least one allow block
        for block in self._allow:
            if block.contains(ip):
                return True
        return False

    def check(self, ip: str) -> dict:
        """Return a detailed verdict dict for *ip*."""
        matched_deny: str | None = None
        for block in self._deny:
            if block.contains(ip):
                matched_deny = f"{block.network}/{block.prefix_len}"
                break

        matched_allow: str | None = None
        for block in self._allow:
            if block.contains(ip):
                matched_allow = f"{block.network}/{block.prefix_len}"
                break

        allowed = self.is_allowed(ip)
        return {
            "ip": ip,
            "allowed": allowed,
            "matched_allow": matched_allow,
            "matched_deny": matched_deny,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IP_ALLOWLIST_REGISTRY: dict = {
    "default": IPAllowlist,
}
