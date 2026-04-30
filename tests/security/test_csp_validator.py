"""Tests for CSP builder."""

from __future__ import annotations

from urllib.parse import urlsplit

from src.security.csp_validator import CSPBuilder, CSPDirective


class TestCSPBuilder:
    def test_strict_default_has_directives(self):
        csp = CSPBuilder.strict_default()
        header = csp.to_header()
        assert "Content-Security-Policy" in header
        assert "default-src" in header["Content-Security-Policy"]

    def test_custom_directive(self):
        csp = CSPBuilder()
        csp.add(CSPDirective("default-src", ["'self'"]))
        assert "default-src 'self'" in csp.build()

    def test_multiple_sources(self):
        csp = CSPBuilder()
        csp.add(CSPDirective("img-src", ["'self'", "https://images.example.com"]))
        parsed = urlsplit(csp.directives[-1].sources[-1])
        assert parsed.scheme == "https"
        assert parsed.hostname == "images.example.com"
        assert f"{parsed.scheme}://{parsed.hostname}" == "https://images.example.com"
