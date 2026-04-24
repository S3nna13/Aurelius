"""Bind-address gate — require explicit opt-in to bind on any-interface.

bandit's B104 reports code that binds to ``0.0.0.0`` but doesn't enforce
anything. This gate raises unless the caller explicitly opts in (kwarg or
``AURELIUS_ALLOW_ANY_INTERFACE=1`` env var). Loopback is always allowed;
non-loopback specific addresses are allowed but logged at WARNING.

Finding AUR-SEC-2026-0023 (typosquat), AUR-SEC-2026-0024 (bind address gate);
CWE-494 (download of code without integrity check),
CWE-1327 (binding to unrestricted IP).

stdlib-only.
"""
from __future__ import annotations

import ipaddress
import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

_ANY_INTERFACE_SENTINELS: Final[frozenset[str]] = frozenset({"0.0.0.0", "::", "*"})  # noqa: S104  # defensive sentinel list — we REJECT these, not bind
_LOOPBACK_NAMES: Final[frozenset[str]] = frozenset({"localhost"})
_OPT_IN_ENV: Final[str] = "AURELIUS_ALLOW_ANY_INTERFACE"


class UnsafeBindAddressError(RuntimeError):
    """Raised when code tries to bind to any-interface without explicit opt-in."""


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    # IPv4-mapped IPv6 — ipaddress keeps them as IPv6Address; unwrap.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return False


def _is_any_interface(host: str) -> bool:
    if host in _ANY_INTERFACE_SENTINELS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_unspecified:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_unspecified
    return False


def _is_valid_host(host: str) -> bool:
    if host in _ANY_INTERFACE_SENTINELS or host in _LOOPBACK_NAMES:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def check_bind_address(host: str, *, allow_any_interface: bool = False) -> str:
    """Validate a bind host, enforcing explicit opt-in for any-interface.

    Args:
        host: The bind address (e.g. ``"127.0.0.1"``, ``"0.0.0.0"``, ``"::"``).
        allow_any_interface: Explicit kwarg opt-in for any-interface binding.
            Also honored via the ``AURELIUS_ALLOW_ANY_INTERFACE=1`` env var.

    Returns:
        The host string unchanged when the check passes.

    Raises:
        ValueError: If ``host`` is empty or not a recognised hostname/IP.
        UnsafeBindAddressError: If ``host`` is any-interface and no opt-in was
            given. The message tells the caller how to opt in.
    """
    if not host:
        raise ValueError("host must be a non-empty string")

    if not _is_valid_host(host):
        raise ValueError(f"Malformed bind address: {host!r}")

    env_opt_in = os.environ.get(_OPT_IN_ENV, "") == "1"

    if _is_any_interface(host):
        if not (allow_any_interface or env_opt_in):
            raise UnsafeBindAddressError(
                f"Refusing to bind on any-interface address {host!r} without "
                f"explicit opt-in. Set env var {_OPT_IN_ENV}=1 or pass "
                f"allow_any_interface=True to proceed. "
                f"(CWE-1327; AUR-SEC-2026-0024)"
            )
        logger.warning(
            "Binding to any-interface address %s (opt-in honored)", host
        )
        return host

    if not _is_loopback(host):
        logger.warning("Binding to non-loopback address %s", host)

    return host


# ---------------------------------------------------------------------------
# Registry (additive)

BIND_ADDRESS_REGISTRY: dict[str, object] = {
    "check": check_bind_address,
    "env_var": _OPT_IN_ENV,
    "any_interface_sentinels": _ANY_INTERFACE_SENTINELS,
}


__all__ = [
    "BIND_ADDRESS_REGISTRY",
    "UnsafeBindAddressError",
    "check_bind_address",
]
