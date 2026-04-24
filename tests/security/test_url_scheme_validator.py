"""Tests for URL scheme validator (AUR-SEC-2026-0020).

Finding AUR-SEC-2026-0020; CWE-918 (SSRF), CWE-284 (improper access control).
"""
from __future__ import annotations

import pytest

from src.security.url_scheme_validator import (
    UnsafeURLSchemeError,
    validate_url,
)


class TestPermittedSchemes:
    def test_accepts_http(self) -> None:
        assert validate_url("http://example.com/path") == "http://example.com/path"

    def test_accepts_https(self) -> None:
        assert validate_url("https://example.com/path") == "https://example.com/path"

    def test_scheme_case_insensitive(self) -> None:
        assert validate_url("HTTPS://Example.com") == "HTTPS://Example.com"
        assert validate_url("HtTp://Example.com") == "HtTp://Example.com"

    def test_custom_allowed_schemes(self) -> None:
        assert (
            validate_url("ws://example.com", allowed_schemes=("ws", "wss"))
            == "ws://example.com"
        )


class TestBannedSchemes:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "FILE:///etc/shadow",
            "gopher://evil.example.com/",
            "ftp://anon@example.com/",
            "jar:http://example.com/x.jar!/y",
            "netdoc:///etc/hosts",
            "javascript:alert(1)",
            "data:text/plain;base64,QQ==",
        ],
    )
    def test_rejects_banned_scheme(self, url: str) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url(url)

    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url("mailto:attacker@example.com")

    def test_rejects_empty_scheme(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url("//example.com/path")


class TestMalformedInputs:
    def test_rejects_non_string(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url(None)  # type: ignore[arg-type]

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url("   ")

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url("http://example.com/\n\rHost: evil")


class TestPrivateIPRejection:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://localhost/",
            "http://0.0.0.0/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://10.255.255.255/",
            "http://172.16.0.1/",
            "http://172.20.1.1/",
            "http://172.31.255.254/",
            "http://192.168.1.1/",
        ],
    )
    def test_rejects_private_ip_when_flag_set(self, url: str) -> None:
        with pytest.raises(UnsafeURLSchemeError):
            validate_url(url, allow_private_ips=False)

    def test_accepts_private_ip_when_flag_true(self) -> None:
        # When allow_private_ips=True (default), localhost is permitted.
        assert (
            validate_url("http://127.0.0.1/", allow_private_ips=True)
            == "http://127.0.0.1/"
        )

    def test_accepts_public_ip_when_flag_false(self) -> None:
        assert (
            validate_url("http://8.8.8.8/", allow_private_ips=False)
            == "http://8.8.8.8/"
        )

    def test_rejects_public_ip_in_172_outside_private_range(self) -> None:
        # 172.15.x.x and 172.32.x.x are NOT in the private 172.16-31 range.
        assert (
            validate_url("http://172.15.0.1/", allow_private_ips=False)
            == "http://172.15.0.1/"
        )
        assert (
            validate_url("http://172.32.0.1/", allow_private_ips=False)
            == "http://172.32.0.1/"
        )


class TestErrorType:
    def test_error_subclass_of_valueerror(self) -> None:
        assert issubclass(UnsafeURLSchemeError, ValueError)

    def test_error_message_includes_scheme(self) -> None:
        with pytest.raises(UnsafeURLSchemeError) as ei:
            validate_url("file:///etc/passwd")
        assert "file" in str(ei.value).lower()


class TestIDN:
    def test_accepts_idn_host(self) -> None:
        # Unicode hostname should be permitted as long as scheme is safe.
        url = "https://xn--e1afmkfd.xn--p1ai/"
        assert validate_url(url) == url

    def test_accepts_raw_unicode_host(self) -> None:
        url = "https://пример.рф/"
        assert validate_url(url) == url
