"""CORS middleware for the stdlib HTTPServer.

Simple middleware that adds CORS headers to all responses
and handles CORS preflight (OPTIONS) requests.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler


class CORSMiddleware:
    """Add CORS headers to all responses and handle OPTIONS preflight."""

    def __init__(
        self,
        allowed_origins: str | list[str] = "*",
        allowed_methods: str = "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        allowed_headers: str = "Content-Type, Authorization, X-API-Key, X-Request-ID",
        allow_credentials: bool = True,
        max_age: int = 86400,
    ) -> None:
        self.allowed_origins = allowed_origins if isinstance(allowed_origins, str) else ", ".join(allowed_origins)
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    def add_headers(self, handler: BaseHTTPRequestHandler) -> None:
        """Add CORS headers to the handler's response."""
        handler.send_header("Access-Control-Allow-Origin", self.allowed_origins)
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
