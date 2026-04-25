"""Web fetch tool for the Aurelius agent surface.

Security-first design: URL allowlist, response size cap, no cookie persistence,
no credential forwarding. Inspired by OpenDevin browser tool
(OpenDevin/OpenDevin, Apache-2.0). License: MIT.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from urllib.parse import urlparse
from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

_MAX_RESPONSE_BYTES = 500_000    # 500KB
_MAX_URL_LEN = 2048
_REQUEST_TIMEOUT = 10            # seconds
_ALLOWED_SCHEMES = frozenset(["https", "http"])

_DENY_HOST_PATTERNS: tuple[re.Pattern, ...] = tuple(re.compile(p) for p in [
    r"^169\.254\.",             # link-local
    r"^127\.",                  # loopback
    r"^10\.",                   # RFC1918
    r"^172\.(1[6-9]|2\d|3[01])\.",  # RFC1918
    r"^192\.168\.",             # RFC1918
    r"^::1$",                   # IPv6 loopback
    r"^fd",                     # IPv6 ULA
    r"localhost$",
    r"metadata\.google\.internal",
    r"169\.254\.169\.254",      # AWS/GCP IMDS
])


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Rejects SSRF-prone targets."""
    if len(url) > _MAX_URL_LEN:
        return False, f"URL exceeds {_MAX_URL_LEN} chars"
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "malformed URL"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme {parsed.scheme!r} not allowed"
    host = parsed.hostname or ""
    for pat in _DENY_HOST_PATTERNS:
        if pat.search(host):
            return False, f"host {host!r} is a denied target (SSRF prevention)"
    return True, ""


class WebTool:
    def fetch(self, url: str, timeout: int = _REQUEST_TIMEOUT) -> ToolResult:
        """Fetch a URL, returning up to _MAX_RESPONSE_BYTES of content."""
        safe, reason = _is_safe_url(url)
        if not safe:
            return ToolResult(tool_name="web", success=False, output="", error=reason)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Aurelius/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read(_MAX_RESPONSE_BYTES)
                content = raw.decode("utf-8", errors="replace")
            return ToolResult(tool_name="web", success=True, output=content, error="")
        except urllib.error.URLError as e:
            return ToolResult(tool_name="web", success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(tool_name="web", success=False, output="", error=str(e))

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web",
            description="Fetch content from a URL",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
            required=["url"],
        )


WEB_TOOL = WebTool()
TOOL_REGISTRY.register(WEB_TOOL.spec(), handler=WEB_TOOL.fetch)
