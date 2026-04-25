"""Tests for CORS validator."""
from __future__ import annotations

import pytest

from src.security.cors_validator import CORSValidator, CORSPolicy


class TestCORSValidator:
    def test_default_allows_all(self):
        cv = CORSValidator()
        assert cv.check_origin("http://evil.com") is True

    def test_restrictive_policy(self):
        cv = CORSValidator()
        cv.add_policy("/api", CORSPolicy(allowed_origins=["https://app.example.com"]))
        assert cv.check_origin("https://app.example.com", "/api") is True
        assert cv.check_origin("http://evil.com", "/api") is False

    def test_method_check(self):
        cv = CORSValidator()
        cv.add_policy("/api", CORSPolicy(allowed_methods=["GET"]))
        assert cv.check_method("GET", "/api") is True
        assert cv.check_method("POST", "/api") is False

    def test_to_headers(self):
        cv = CORSValidator()
        cv.add_policy("/api", CORSPolicy(
            allowed_origins=["*"], allowed_methods=["GET"], allow_credentials=True,
        ))
        headers = cv.to_headers("/api")
        assert "Access-Control-Allow-Origin" in headers
        assert "Access-Control-Allow-Credentials" in headers