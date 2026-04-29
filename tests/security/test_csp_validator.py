"""Tests for CSP builder."""

from __future__ import annotations

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
        assert "https://images.example.com" in csp.directives[-1].sources
