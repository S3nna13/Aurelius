"""Simple HTTP client wrapper for agent tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_DENY_HOST_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"^169\.254\.",
        r"^127\.",
        r"^10\.",
        r"^172\.(1[6-9]|2\d|3[01])\.",
        r"^192\.168\.",
        r"^::1$",
        r"^fd",
        r"localhost$",
        r"metadata\.google\.internal",
        r"169\.254\.169\.254",
    ]
)

_ALLOWED_SCHEMES = frozenset(["https", "http"])


def _is_safe_url(url: str) -> tuple[bool, str]:
    if len(url) > 2048:
        return False, "URL exceeds maximum length"
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "malformed URL"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme {parsed.scheme!r} not allowed"
    host = parsed.hostname or ""
    for pat in _DENY_HOST_PATTERNS:
        if pat.search(host):
            return False, f"host {host!r} blocked (SSRF prevention)"
    return True, ""


@dataclass
class HTTPResponse:
    status: int
    body: str
    headers: dict[str, str] | None = None


@dataclass
class SimpleHTTPClient:
    """Minimal HTTP client for agent tool use."""

    timeout_seconds: float = 30.0

    def get(self, url: str) -> HTTPResponse:
        safe, reason = _is_safe_url(url)
        if not safe:
            return HTTPResponse(status=0, body=reason)

        from urllib.request import Request, urlopen

        req = Request(url, method="GET")  # noqa: S310  # nosec
        try:
            resp = urlopen(req, timeout=self.timeout_seconds)  # noqa: S310  # nosec
            body = resp.read().decode("utf-8", errors="replace")
            return HTTPResponse(status=resp.status, body=body)
        except Exception as e:
            return HTTPResponse(status=0, body=str(e))

    def post(self, url: str, data: dict[str, Any]) -> HTTPResponse:
        safe, reason = _is_safe_url(url)
        if not safe:
            return HTTPResponse(status=0, body=reason)

        import json
        from urllib.request import Request, urlopen

        payload = json.dumps(data).encode("utf-8")
        req = Request(url, data=payload, method="POST")  # noqa: S310  # nosec
        req.add_header("Content-Type", "application/json")
        try:
            resp = urlopen(req, timeout=self.timeout_seconds)  # noqa: S310  # nosec
            body = resp.read().decode("utf-8", errors="replace")
            return HTTPResponse(status=resp.status, body=body)
        except Exception as e:
            return HTTPResponse(status=0, body=str(e))


HTTP_CLIENT = SimpleHTTPClient()
