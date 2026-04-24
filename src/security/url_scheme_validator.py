"""URL scheme & SSRF validator.

Finding AUR-SEC-2026-0020; CWE-918 (SSRF), CWE-284 (improper access control).

Provides a single entry point, :func:`validate_url`, for gating ``urllib``
call-sites against unsafe schemes (``file:``, ``gopher:``, ``ftp:``, ``jar:``,
``netdoc:``, ``javascript:``, ``data:``) and, optionally, against RFC-1918
private / loopback / link-local destinations commonly abused for SSRF
(cloud metadata, intranet pivoting).

The validator is intentionally *allow-list* based: only schemes the caller
explicitly names may pass. Any parse failure, unknown scheme, or control
character aborts with :class:`UnsafeURLSchemeError` (subclass of
:class:`ValueError`). No silent fallbacks; no network I/O.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Sequence
from urllib.parse import urlsplit

__all__ = [
    "UnsafeURLSchemeError",
    "validate_url",
    "BANNED_SCHEMES",
]

_LOG = logging.getLogger(__name__)

#: Schemes known to be abused in SSRF / local-file-read attacks.
BANNED_SCHEMES: frozenset[str] = frozenset(
    {
        "file",
        "gopher",
        "ftp",
        "jar",
        "netdoc",
        "javascript",
        "data",
    }
)


class UnsafeURLSchemeError(ValueError):
    """Raised when a URL fails scheme or SSRF validation."""


def _has_control_chars(s: str) -> bool:
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in s)


def _is_private_host(host: str) -> bool:
    """Return True when *host* resolves to a private / loopback / link-local IP.

    Hostname lookups deliberately avoid DNS — a real deployment that wants to
    block DNS-rebind SSRF should additionally resolve and re-validate at
    connect-time. Here we cover the common literal cases: ``localhost``,
    ``0.0.0.0``, and any IPv4/IPv6 address the ``ipaddress`` module recognises
    as private, loopback, or link-local.
    """
    if not host:
        return False
    bare = host.lower().strip("[]")
    if bare in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True
    # Strip zone id (IPv6 scope) if present.
    bare_noscope = bare.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(bare_noscope)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
    )


def validate_url(
    url: str,
    allowed_schemes: Sequence[str] = ("http", "https"),
    *,
    allow_private_ips: bool = True,
) -> str:
    """Validate *url* and return it unchanged on success.

    Args:
        url: URL to validate.
        allowed_schemes: Case-insensitive whitelist of schemes. Defaults to
            ``("http", "https")``.
        allow_private_ips: When ``False``, reject URLs whose host is a
            loopback, link-local, private (RFC-1918), or unspecified address
            (SSRF defense; blocks cloud-metadata pivots such as
            ``169.254.169.254``).

    Raises:
        UnsafeURLSchemeError: on any validation failure. The error is logged
            at WARNING level before being raised (no silent swallowing).
    """
    if not isinstance(url, str):
        msg = f"URL must be a string, got {type(url).__name__}"
        _LOG.warning("validate_url rejected non-string: %r", url)
        raise UnsafeURLSchemeError(msg)

    if not url or not url.strip():
        _LOG.warning("validate_url rejected empty URL")
        raise UnsafeURLSchemeError("URL must be a non-empty string")

    if _has_control_chars(url):
        _LOG.warning("validate_url rejected URL with control chars: %r", url)
        raise UnsafeURLSchemeError("URL contains control characters")

    try:
        parts = urlsplit(url)
    except ValueError as exc:
        _LOG.warning("validate_url rejected unparseable URL %r: %s", url, exc)
        raise UnsafeURLSchemeError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if not scheme:
        _LOG.warning("validate_url rejected URL without scheme: %r", url)
        raise UnsafeURLSchemeError(f"URL is missing a scheme: {url!r}")

    if scheme in BANNED_SCHEMES:
        _LOG.warning("validate_url rejected banned scheme %r in %r", scheme, url)
        raise UnsafeURLSchemeError(
            f"URL scheme {scheme!r} is not permitted (banned)"
        )

    allow_norm = {s.lower() for s in allowed_schemes}
    if scheme not in allow_norm:
        _LOG.warning(
            "validate_url rejected scheme %r (allowed: %s)", scheme, sorted(allow_norm)
        )
        raise UnsafeURLSchemeError(
            f"URL scheme {scheme!r} not in allowed schemes {sorted(allow_norm)}"
        )

    if not allow_private_ips:
        host = parts.hostname or ""
        if _is_private_host(host):
            _LOG.warning(
                "validate_url rejected private/loopback host %r in %r", host, url
            )
            raise UnsafeURLSchemeError(
                f"URL host {host!r} resolves to a private/loopback/link-local "
                "address; blocked for SSRF"
            )

    return url


# Suppress unused-import lint warnings; socket is intentionally available for
# future DNS-based SSRF hardening.
_ = socket
