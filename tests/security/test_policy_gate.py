"""Tests for src.security.policy_gate."""

from __future__ import annotations

import logging

import pytest

from src.security.policy_gate import (
    DEFAULT_POLICY_GATE,
    POLICY_GATE_REGISTRY,
    PolicyGate,
    PolicyScope,
    PolicyViolation,
)


def _make_gate() -> PolicyGate:
    return PolicyGate()


def test_default_deny_no_policy_raises():
    gate = _make_gate()
    with pytest.raises(PolicyViolation) as excinfo:
        gate.check("example.com", "scan")
    assert excinfo.value.action == "scan"
    assert excinfo.value.reason == "no matching policy"
    assert excinfo.value.scope is None


def test_authorized_action_returns_true():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="example.com",
            allowed_actions=frozenset({"scan", "enumerate"}),
            owner="secops",
        )
    )
    assert gate.check("example.com", "scan") is True
    assert gate.check("example.com", "enumerate") is True


def test_is_authorized_false_when_no_match():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="example.com",
            allowed_actions=frozenset({"scan"}),
            owner="secops",
        )
    )
    assert gate.is_authorized("other.com", "scan") is False
    assert gate.is_authorized("example.com", "exploit") is False


def test_is_authorized_true_when_match():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="example.com",
            allowed_actions=frozenset({"scan"}),
            owner="secops",
        )
    )
    assert gate.is_authorized("example.com", "scan") is True


def test_multiple_policies_independent():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="a.example",
            allowed_actions=frozenset({"scan"}),
            owner="team-a",
        )
    )
    gate.add_policy(
        PolicyScope(
            target_domain="b.example",
            allowed_actions=frozenset({"enumerate"}),
            owner="team-b",
        )
    )
    assert gate.is_authorized("a.example", "scan") is True
    assert gate.is_authorized("b.example", "enumerate") is True
    # Cross-policy leakage must not happen.
    assert gate.is_authorized("a.example", "enumerate") is False
    assert gate.is_authorized("b.example", "scan") is False
    assert len(gate.list_policies()) == 2


def test_empty_action_rejected():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="example.com",
            allowed_actions=frozenset({"scan"}),
            owner="secops",
        )
    )
    with pytest.raises(ValueError):
        gate.check("example.com", "")
    with pytest.raises(ValueError):
        gate.is_authorized("example.com", "")


def test_target_must_be_string():
    gate = _make_gate()
    with pytest.raises(TypeError):
        gate.check(123, "scan")  # type: ignore[arg-type]


def test_wildcard_scope_matches_any_target():
    gate = _make_gate()
    gate.add_policy(
        PolicyScope(
            target_domain="*",
            allowed_actions=frozenset({"observe"}),
            owner="platform",
        )
    )
    assert gate.is_authorized("anything.com", "observe") is True
    assert gate.is_authorized("other.com", "observe") is True
    assert gate.is_authorized("anything.com", "exploit") is False


def test_denial_logged_at_warning(caplog):
    gate = _make_gate()
    with caplog.at_level(logging.WARNING, logger="aurelius.security.policy_gate"):
        with pytest.raises(PolicyViolation):
            gate.check("example.com", "scan")
    assert any("policy_gate_denied" in rec.getMessage() for rec in caplog.records)


def test_list_policies_returns_copy():
    gate = _make_gate()
    scope = PolicyScope(
        target_domain="example.com",
        allowed_actions=frozenset({"scan"}),
        owner="secops",
    )
    gate.add_policy(scope)
    listed = gate.list_policies()
    listed.clear()
    assert len(gate.list_policies()) == 1


def test_default_registry_present():
    assert "default" in POLICY_GATE_REGISTRY
    assert POLICY_GATE_REGISTRY["default"] is DEFAULT_POLICY_GATE


def test_policy_scope_rejects_empty_target():
    with pytest.raises(ValueError):
        PolicyScope(target_domain="", allowed_actions=frozenset(), owner="x")


def test_policy_scope_rejects_bad_action_entry():
    with pytest.raises(ValueError):
        PolicyScope(
            target_domain="example.com",
            allowed_actions=frozenset({""}),
            owner="x",
        )


def test_add_policy_type_check():
    gate = _make_gate()
    with pytest.raises(TypeError):
        gate.add_policy("not-a-scope")  # type: ignore[arg-type]
