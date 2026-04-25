"""Simple HTTP client wrapper for agent tool calls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        from urllib.request import urlopen, Request
        req = Request(url, method="GET")
        try:
            resp = urlopen(req, timeout=self.timeout_seconds)
            body = resp.read().decode("utf-8", errors="replace")
            return HTTPResponse(status=resp.status, body=body)
        except Exception as e:
            return HTTPResponse(status=0, body=str(e))

    def post(self, url: str, data: dict[str, Any]) -> HTTPResponse:
        import json
        from urllib.request import urlopen, Request
        payload = json.dumps(data).encode("utf-8")
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            resp = urlopen(req, timeout=self.timeout_seconds)
            body = resp.read().decode("utf-8", errors="replace")
            return HTTPResponse(status=resp.status, body=body)
        except Exception as e:
            return HTTPResponse(status=0, body=str(e))


HTTP_CLIENT = SimpleHTTPClient()