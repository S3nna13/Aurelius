"""Tests for HTTP client."""
from __future__ import annotations

import pytest

from src.tools.http_client import SimpleHTTPClient, HTTPResponse


class TestHTTPResponse:
    def test_response_fields(self):
        r = HTTPResponse(status=200, body="ok", headers={"x-test": "1"})
        assert r.status == 200
        assert r.body == "ok"
        assert r.headers["x-test"] == "1"

    def test_response_default_headers(self):
        r = HTTPResponse(status=500, body="error")
        assert r.headers is None


class TestSimpleHTTPClient:
    def test_get_nonexistent_returns_error(self):
        client = SimpleHTTPClient(timeout_seconds=5.0)
        resp = client.get("http://nonexistent.example.test/")
        assert resp.status == 0
        assert len(resp.body) > 0