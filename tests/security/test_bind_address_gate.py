"""Tests for src.security.bind_address_gate (AUR-SEC-2026-0024)."""
from __future__ import annotations

import logging
import os

import pytest

from src.security.bind_address_gate import (
    UnsafeBindAddressError,
    check_bind_address,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURELIUS_ALLOW_ANY_INTERFACE", raising=False)


def test_ipv4_loopback_allowed() -> None:
    assert check_bind_address("127.0.0.1") == "127.0.0.1"


def test_ipv6_loopback_allowed() -> None:
    assert check_bind_address("::1") == "::1"


def test_localhost_allowed() -> None:
    assert check_bind_address("localhost") == "localhost"


def test_zero_zero_rejected() -> None:
    with pytest.raises(UnsafeBindAddressError) as exc_info:
        check_bind_address("0.0.0.0")
    msg = str(exc_info.value)
    assert "AURELIUS_ALLOW_ANY_INTERFACE" in msg
    assert "allow_any_interface" in msg


def test_ipv6_any_rejected() -> None:
    with pytest.raises(UnsafeBindAddressError):
        check_bind_address("::")


def test_star_rejected() -> None:
    with pytest.raises(UnsafeBindAddressError):
        check_bind_address("*")


def test_kwarg_opt_in_allows_any() -> None:
    assert check_bind_address("0.0.0.0", allow_any_interface=True) == "0.0.0.0"
    assert check_bind_address("::", allow_any_interface=True) == "::"


def test_env_var_opt_in_allows_any(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURELIUS_ALLOW_ANY_INTERFACE", "1")
    assert check_bind_address("0.0.0.0") == "0.0.0.0"


def test_env_var_zero_does_not_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURELIUS_ALLOW_ANY_INTERFACE", "0")
    with pytest.raises(UnsafeBindAddressError):
        check_bind_address("0.0.0.0")


def test_non_loopback_specific_ip_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.security.bind_address_gate"):
        result = check_bind_address("10.0.0.5")
    assert result == "10.0.0.5"
    assert any("10.0.0.5" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records)


def test_loopback_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.security.bind_address_gate"):
        check_bind_address("127.0.0.1")
    assert not any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_malformed_address_rejected() -> None:
    with pytest.raises(ValueError):
        check_bind_address("not an address")


def test_empty_string_rejected() -> None:
    with pytest.raises(ValueError):
        check_bind_address("")


def test_ipv4_mapped_ipv6_loopback_allowed() -> None:
    # ::ffff:127.0.0.1 should be treated as loopback
    assert check_bind_address("::ffff:127.0.0.1") == "::ffff:127.0.0.1"


def test_ipv4_mapped_ipv6_any_rejected() -> None:
    # ::ffff:0.0.0.0 maps to 0.0.0.0 → must reject
    with pytest.raises(UnsafeBindAddressError):
        check_bind_address("::ffff:0.0.0.0")


def test_returns_host_unchanged_on_success() -> None:
    assert check_bind_address("127.0.0.1") == "127.0.0.1"
    assert check_bind_address("0.0.0.0", allow_any_interface=True) == "0.0.0.0"
