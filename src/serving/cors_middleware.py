"""CORS middleware for the stdlib HTTPServer.

Simple middleware that adds CORS headers to all responses
and handles CORS preflight (OPTIONS) requests.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit


def _normalize_origin(origin: str) -> str | None:
    """Return a canonical origin string or ``None`` for invalid input."""
    origin = origin.strip()
    if not origin or "\r" in origin or "\n" in origin:
        return None

    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.path or parsed.query or parsed.fragment or "@" in parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


class CORSMiddleware:
    """Add CORS headers to all responses and handle OPTIONS preflight."""

    def __init__(
        self,
        allowed_origins: str | list[str] = "http://localhost:5173",
        allowed_methods: str = "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        allowed_headers: str = "Content-Type, Authorization, X-API-Key, X-Request-ID",
        allow_credentials: bool = False,
        max_age: int = 86400,
    ) -> None:
        if isinstance(allowed_origins, str):
            origins = [o.strip() for o in allowed_origins.split(",")]
        else:
            origins = list(allowed_origins)
        self._origins = [origin for origin in (_normalize_origin(o) for o in origins) if origin]
        self._fallback_origin = self._origins[0] if self._origins else ""
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    def add_headers(self, handler: BaseHTTPRequestHandler) -> None:
        """Add CORS headers to the handler's response."""
        origin = handler.headers.get("Origin", "")
        normalized = _normalize_origin(origin)
        if normalized in self._origins:
            handler.send_header("Access-Control-Allow-Origin", normalized)
        elif self._fallback_origin:
            handler.send_header("Access-Control-Allow-Origin", self._fallback_origin)
        handler.send_header("Access-Control-Allow-Methods", self.allowed_methods)
        handler.send_header("Access-Control-Allow-Headers", self.allowed_headers)
        if self.allow_credentials:
            handler.send_header("Access-Control-Allow-Credentials", "true")
        handler.send_header("Access-Control-Max-Age", str(self.max_age))

    def handle_preflight(self, handler: BaseHTTPRequestHandler) -> bool:
        """Handle CORS preflight OPTIONS request.

        Returns True if the request was handled (OPTIONS), False otherwise.
        """
        if handler.command != "OPTIONS":
            return False

        handler.send_response(204)
        self.add_headers(handler)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return True


CORS = CORSMiddleware()
