"""Structured web-browse / web-fetch tool descriptors for the Aurelius agent.

This module defines the **schema** and **local validator** for a safe
web-fetch tool the agent loop can invoke. It deliberately does **not**
make real HTTP requests — the agent runtime wires an actual fetcher in a
separate, higher-privilege layer. Here we only:

  * Validate URLs (scheme, host, private IP ranges).
  * Build `WebRequestSpec` dataclasses the runtime consumes.
  * Produce a JSON-schema-style tool descriptor for registration in
    ``TOOL_REGISTRY``.
  * Summarise a `WebFetchResult` for downstream agent consumption.

Stdlib only: ``dataclasses``, ``ipaddress``, ``re``, ``urllib.parse``.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_TOOL_DESCRIPTOR",
    "PrivateHostBlocked",
    "UrlValidationError",
    "WebBrowseTool",
    "WebFetchResult",
    "WebRequestSpec",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UrlValidationError(ValueError):
    """Raised when a URL or request spec fails structural validation."""


class PrivateHostBlocked(UrlValidationError):
    """Raised when a URL targets a private / loopback / link-local host."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_ALLOWED_METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE"}
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Private / reserved IPv4 ranges that must never be reachable from the
# agent. Matches the spec: RFC1918 + loopback + link-local.
_PRIVATE_V4_NETS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
)

# Hostnames that always resolve to loopback and so must be refused
# without any DNS interaction.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)

# Cheap sanity check for IPv4 literal form; avoids calling ipaddress on
# every hostname.
_IPV4_LITERAL_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ---------------------------------------------------------------------------
# Host classification
# ---------------------------------------------------------------------------


def _host_is_private(host: str) -> bool:
    """Return True if *host* is a literal private/loopback/link-local address.

    Hostnames that are not IP literals are only classified as blocked when
    they appear in the static ``_BLOCKED_HOSTNAMES`` set — DNS resolution
    is intentionally not performed here (the module is side-effect free).
    """
    if not host:
        return True  # empty host is never valid

    lowered = host.lower()
    # Strip brackets from IPv6 literals: "[::1]" -> "::1"
    if lowered.startswith("[") and lowered.endswith("]"):
        lowered = lowered[1:-1]

    if lowered in _BLOCKED_HOSTNAMES:
        return True

    # Try IPv4 literal
    if _IPV4_LITERAL_RE.match(lowered):
        try:
            ip = ipaddress.IPv4Address(lowered)
        except ValueError:
            return False
        for net in _PRIVATE_V4_NETS:
            if ip in net:
                return True
        return False

    # Try IPv6 literal
    if ":" in lowered:
        try:
            ip6 = ipaddress.IPv6Address(lowered)
        except ValueError:
            return False
        if ip6.is_loopback or ip6.is_link_local or ip6.is_private:
            return True
        # Also catch IPv4-mapped forms like ::ffff:127.0.0.1
        if ip6.ipv4_mapped is not None:
            for net in _PRIVATE_V4_NETS:
                if ip6.ipv4_mapped in net:
                    return True
        return False

    return False


def _validate_url(url: str) -> tuple[str, str]:
    """Validate *url* and return ``(scheme, host)``.

    Raises :class:`UrlValidationError` for malformed or non-http(s) URLs
    and :class:`PrivateHostBlocked` for private/loopback hosts.
    """
    if not isinstance(url, str) or not url:
        raise UrlValidationError("url must be a non-empty string")

    # Fast prefix check — matches spec wording "starts with http:// or https://"
    low = url.lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        raise UrlValidationError(f"url scheme must be http or https: {url!r}")

    try:
        parts = urlsplit(url)
    except ValueError as exc:  # pragma: no cover - urlsplit rarely raises
        raise UrlValidationError(f"malformed url: {url!r}") from exc

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UrlValidationError(f"url scheme must be http or https: {url!r}")

    host = parts.hostname or ""
    if not host:
        raise UrlValidationError(f"url missing host: {url!r}")

    # IDN: attempt ASCII/punycode encoding. If it fails, reject cleanly
    # rather than propagating a surprise UnicodeError at request time.
    try:
        host.encode("idna")
    except UnicodeError as exc:
        raise UrlValidationError(f"invalid internationalised host: {host!r}") from exc

    if _host_is_private(host):
        raise PrivateHostBlocked(f"refusing private/loopback host: {host!r}")

    return scheme, host


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WebRequestSpec:
    """Declarative spec for a single outbound web request.

    This is a *plan*, not an execution. Validation happens in
    ``__post_init__``; the agent runtime translates validated specs into
    actual HTTP calls elsewhere.
    """

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 10.0
    max_bytes: int = 2_000_000
    follow_redirects: bool = True
    user_agent: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str):
            raise UrlValidationError("method must be a string")
        self.method = self.method.upper()
        if self.method not in _ALLOWED_METHODS:
            raise UrlValidationError(
                f"method must be one of {sorted(_ALLOWED_METHODS)}: {self.method!r}"
            )

        # URL + host validation (may raise PrivateHostBlocked)
        _validate_url(self.url)

        if not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            raise UrlValidationError("timeout_s must be > 0")
        self.timeout_s = float(self.timeout_s)

        if not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise UrlValidationError("max_bytes must be a positive int")

        if self.headers is None:
            self.headers = {}
        if not isinstance(self.headers, dict):
            raise UrlValidationError("headers must be a dict")
        # Enforce str/str headers for determinism.
        for k, v in self.headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise UrlValidationError("headers must map str -> str")

        if self.user_agent is not None and not isinstance(self.user_agent, str):
            raise UrlValidationError("user_agent must be str or None")

        # If a user agent is set and not already present in headers,
        # surface it there so downstream consumers see a canonical form.
        if self.user_agent and "User-Agent" not in self.headers:
            self.headers["User-Agent"] = self.user_agent


@dataclass
class WebFetchResult:
    """Result of an executed web request, produced by the runtime fetcher."""

    status: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body_sample: str = ""
    bytes_read: int = 0
    elapsed_s: float = 0.0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class WebBrowseTool:
    """Builder + validator for agent-issued web-fetch calls.

    Does not make network calls. The agent runtime is responsible for
    executing validated :class:`WebRequestSpec` instances.
    """

    def __init__(
        self,
        default_timeout_s: float = 10.0,
        default_max_bytes: int = 2_000_000,
        default_user_agent: str | None = "Aurelius-Agent/1.0",
    ) -> None:
        if default_timeout_s <= 0:
            raise UrlValidationError("default_timeout_s must be > 0")
        if default_max_bytes <= 0:
            raise UrlValidationError("default_max_bytes must be > 0")
        self.default_timeout_s = float(default_timeout_s)
        self.default_max_bytes = int(default_max_bytes)
        self.default_user_agent = default_user_agent

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def build_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        **overrides: Any,
    ) -> WebRequestSpec:
        """Build and validate a :class:`WebRequestSpec`.

        ``overrides`` may set any of ``timeout_s``, ``max_bytes``,
        ``follow_redirects``, ``user_agent``.
        """
        spec_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": dict(headers) if headers else {},
            "timeout_s": overrides.pop("timeout_s", self.default_timeout_s),
            "max_bytes": overrides.pop("max_bytes", self.default_max_bytes),
            "follow_redirects": overrides.pop("follow_redirects", True),
            "user_agent": overrides.pop("user_agent", self.default_user_agent),
        }
        if overrides:
            raise UrlValidationError(
                f"unknown overrides: {sorted(overrides)!r}"
            )
        return WebRequestSpec(**spec_kwargs)

    def validate_request(self, spec: WebRequestSpec) -> None:
        """Re-run validation on an already-constructed spec."""
        if not isinstance(spec, WebRequestSpec):
            raise UrlValidationError("spec must be a WebRequestSpec")
        # Re-trigger by building a fresh instance from spec fields.
        WebRequestSpec(
            method=spec.method,
            url=spec.url,
            headers=dict(spec.headers),
            timeout_s=spec.timeout_s,
            max_bytes=spec.max_bytes,
            follow_redirects=spec.follow_redirects,
            user_agent=spec.user_agent,
        )

    # ------------------------------------------------------------------
    # Result summarisation
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_result(result: WebFetchResult, max_chars: int = 2000) -> str:
        """Produce a compact textual summary of a fetch result.

        Deterministic and stdlib-only so the agent loop can feed it back
        as an observation without surprises.
        """
        if not isinstance(result, WebFetchResult):
            raise UrlValidationError("result must be a WebFetchResult")
        if max_chars <= 0:
            raise UrlValidationError("max_chars must be > 0")

        body = result.body_sample or ""
        truncated = result.truncated
        if len(body) > max_chars:
            body = body[:max_chars]
            truncated = True

        header = (
            f"HTTP {result.status} {result.url} "
            f"({result.bytes_read} bytes, {result.elapsed_s:.3f}s"
            f"{', truncated' if truncated else ''})"
        )
        if body:
            return f"{header}\n{body}"
        return header


# ---------------------------------------------------------------------------
# Tool descriptor (JSON-schema style) for TOOL_REGISTRY
# ---------------------------------------------------------------------------


DEFAULT_TOOL_DESCRIPTOR: dict[str, Any] = {
    "name": "web_browse",
    "description": (
        "Fetch a single HTTP(S) URL and return a bounded body sample. "
        "Private/loopback hosts are refused. Does not execute JavaScript."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http:// or https:// URL to fetch.",
            },
            "method": {
                "type": "string",
                "enum": sorted(_ALLOWED_METHODS),
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "default": {},
            },
            "timeout_s": {
                "type": "number",
                "minimum": 0,
                "exclusiveMinimum": True,
                "default": 10.0,
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "default": 2_000_000,
            },
            "follow_redirects": {"type": "boolean", "default": True},
            "user_agent": {"type": ["string", "null"], "default": "Aurelius-Agent/1.0"},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}
