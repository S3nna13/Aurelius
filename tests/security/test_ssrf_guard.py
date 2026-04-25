"""Regression tests for AUR-SEC-2026-0020 SSRF guard (B310 / CWE-22).

Verifies that the URL validator rejects file:// schemes, RFC1918 private
addresses and other SSRF-prone hosts, and accepts well-formed http/https
URLs pointing at public hosts.
"""
from __future__ import annotations

import pytest

from src.tools.web_tool import _is_safe_url


def test_validator_blocks_file_scheme():
    safe, reason = _is_safe_url("file:///etc/passwd")
    assert safe is False
    assert "file" in reason or "scheme" in reason


def test_validator_blocks_rfc1918_192_168():
    safe, reason = _is_safe_url("http://192.168.1.1/secret")
    assert safe is False
    assert "SSRF" in reason or "denied" in reason


def test_validator_blocks_rfc1918_10():
    safe, _ = _is_safe_url("http://10.0.0.5/admin")
    assert safe is False


def test_validator_blocks_rfc1918_172():
    safe, _ = _is_safe_url("http://172.16.5.5/admin")
    assert safe is False


def test_validator_blocks_loopback():
    safe, _ = _is_safe_url("http://127.0.0.1:8080/")
    assert safe is False


def test_validator_blocks_localhost():
    safe, _ = _is_safe_url("http://localhost/")
    assert safe is False


def test_validator_blocks_link_local_imds():
    safe, _ = _is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert safe is False


def test_validator_allows_public_http():
    safe, reason = _is_safe_url("http://example.com/path")
    assert safe is True, reason


def test_validator_allows_public_https():
    safe, reason = _is_safe_url("https://example.com/api/v1/chat")
    assert safe is True, reason


def test_validator_blocks_gopher():
    safe, _ = _is_safe_url("gopher://example.com/")
    assert safe is False


# --- Backend-specific scheme validators -------------------------------------

def test_http_backend_validator_rejects_file_scheme():
    from src.backends.http_backend import _validate_backend_url

    with pytest.raises(ValueError):
        _validate_backend_url("file:///etc/passwd")


def test_http_backend_validator_accepts_http():
    from src.backends.http_backend import _validate_backend_url

    _validate_backend_url("http://localhost:8000/v1/chat/completions")


def test_ollama_validator_rejects_file_scheme():
    from src.backends.ollama_adapter import _validate_ollama_url

    with pytest.raises(ValueError):
        _validate_ollama_url("file:///etc/shadow")


def test_ollama_validator_accepts_http():
    from src.backends.ollama_adapter import _validate_ollama_url

    _validate_ollama_url("http://localhost:11434/api/tags")


def test_web_ui_validator_rejects_file_scheme():
    from src.serving.web_ui import _validate_upstream_url

    with pytest.raises(ValueError):
        _validate_upstream_url("file:///etc/passwd")


def test_web_ui_validator_rejects_gopher():
    from src.serving.web_ui import _validate_upstream_url

    with pytest.raises(ValueError):
        _validate_upstream_url("gopher://example.com/")


def test_web_ui_validator_accepts_https():
    from src.serving.web_ui import _validate_upstream_url

    _validate_upstream_url("https://api.example.com/v1/chat/completions")


# --- AUR-SEC-2026-0023 bind default regression -------------------------------

def test_serve_config_default_host_is_loopback():
    """Default serving bind must be loopback (CWE-605 B104)."""
    from src.deployment.serve_config import ServeDeploymentConfig

    cfg = ServeDeploymentConfig(model_name="m", model_path="/tmp/m")
    assert cfg.host == "127.0.0.1", (
        f"Default host must be 127.0.0.1 not {cfg.host!r} (CWE-605 B104)"
    )
