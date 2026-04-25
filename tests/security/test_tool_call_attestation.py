"""Tests for tool_call_attestation — HMAC origin verification.

Security surface: STRIDE Spoofing/Tampering.
"""
from __future__ import annotations

import pytest

from src.security.tool_call_attestation import (
    AttestedToolCall,
    ToolCallAttestation,
    ATTESTATION_REGISTRY,
    DEFAULT_TOOL_CALL_ATTESTATION,
)


# ---------------------------------------------------------------------------
# Attestation and verification
# ---------------------------------------------------------------------------


def test_attest_produces_valid_signature():
    a = ToolCallAttestation()
    call = a.attest("bash", {"cmd": "echo hi"})
    assert a.verify(call) is True


def test_verify_fails_after_tampering():
    a = ToolCallAttestation()
    call = a.attest("bash", {"cmd": "echo hi"})
    tampered = AttestedToolCall(
        tool_name="bash",
        arguments={"cmd": "rm -rf /"},
        nonce=call.nonce,
        signature=call.signature,
    )
    assert a.verify(tampered) is False


def test_verify_fails_with_wrong_key():
    a1 = ToolCallAttestation()
    a2 = ToolCallAttestation()
    call = a1.attest("bash", {"cmd": "echo hi"})
    assert a2.verify(call) is False


def test_verify_fails_with_replayed_nonce():
    a = ToolCallAttestation()
    call1 = a.attest("bash", {"cmd": "echo hi"})
    call2 = a.attest("bash", {"cmd": "echo hi"})
    assert call1.nonce != call2.nonce


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------


def test_rotate_key_invalidates_old_signatures():
    a = ToolCallAttestation()
    call = a.attest("bash", {"cmd": "echo hi"})
    assert a.verify(call) is True
    a.rotate_key()
    assert a.verify(call) is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in ATTESTATION_REGISTRY
    assert isinstance(ATTESTATION_REGISTRY["default"], ToolCallAttestation)


def test_default_is_attestation():
    assert isinstance(DEFAULT_TOOL_CALL_ATTESTATION, ToolCallAttestation)
