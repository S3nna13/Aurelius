"""Aurelius security policy gate.

A default-deny authorization gate that decides whether a given
``(target, action)`` pair is permitted by any registered policy scope.

Design notes:
    * Fail closed: if no policy matches, the request is DENIED.
    * All denials are routed through the stdlib ``logging`` module at
      WARNING level. We never ``print`` security decisions -- that would
      be capturable by untrusted code redirecting stdout.
    * The gate accepts only strings for ``target``/``action`` and rejects
      empty actions outright. All inputs are considered untrusted.
    * Policies are compared by exact target match OR by the wildcard
      ``"*"`` target, which denotes "any target". This keeps the match
      logic simple and auditable.

This module is 100% original Aurelius code. It was written from the
structural description of a policy gate (scope / check / registry) and
shares no code with any external repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

__all__ = [
    "PolicyScope",
    "PolicyViolation",
    "PolicyGate",
    "POLICY_GATE_REGISTRY",
    "DEFAULT_POLICY_GATE",
]


_LOGGER = logging.getLogger("aurelius.security.policy_gate")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyScope:
    """An authorization scope owned by a principal.

    Attributes:
        target_domain: Exact target string the scope applies to, or the
            literal ``"*"`` to match any target.
        allowed_actions: Immutable set of action names permitted inside
            this scope.
        owner: Human-readable owner string (team / operator / service).
    """

    target_domain: str
    allowed_actions: frozenset[str] = field(default_factory=frozenset)
    owner: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.target_domain, str) or not self.target_domain:
            raise ValueError("PolicyScope.target_domain must be a non-empty string")
        if not isinstance(self.allowed_actions, frozenset):
            # Coerce so callers can pass a regular set/list at construction
            object.__setattr__(self, "allowed_actions", frozenset(self.allowed_actions))
        for action in self.allowed_actions:
            if not isinstance(action, str) or not action:
                raise ValueError("PolicyScope.allowed_actions entries must be non-empty strings")
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("PolicyScope.owner must be a non-empty string")

    def matches_target(self, target: str) -> bool:
        return self.target_domain == "*" or self.target_domain == target

    def permits(self, action: str) -> bool:
        return action in self.allowed_actions


class PolicyViolation(Exception):
    """Raised when a policy check denies an action."""

    def __init__(self, scope: PolicyScope | None, action: str, reason: str) -> None:
        self.scope = scope
        self.action = action
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        scope_repr = (
            f"scope(target={self.scope.target_domain!r}, owner={self.scope.owner!r})"
            if self.scope is not None
            else "scope(<none>)"
        )
        return f"PolicyViolation: action={self.action!r} {scope_repr} reason={self.reason!r}"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class PolicyGate:
    """Authorizes ``(target, action)`` pairs against registered scopes.

    The gate stores policies in insertion order. ``check`` returns ``True``
    when at least one scope matches the target AND permits the action.
    Otherwise a ``PolicyViolation`` is raised.
    """

    def __init__(self) -> None:
        self._policies: list[PolicyScope] = []

    # --- mutation ---------------------------------------------------------

    def add_policy(self, scope: PolicyScope) -> None:
        if not isinstance(scope, PolicyScope):
            raise TypeError("add_policy requires a PolicyScope instance")
        self._policies.append(scope)

    def list_policies(self) -> list[PolicyScope]:
        # Defensive copy so callers cannot mutate internal state.
        return list(self._policies)

    # --- authorization ----------------------------------------------------

    def _validate_inputs(self, target: str, action: str) -> None:
        if not isinstance(target, str):
            raise TypeError("target must be a string")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")

    def check(self, target: str, action: str) -> bool:
        """Return True iff authorized; otherwise raise PolicyViolation.

        Default-deny: with no policies registered, every call raises.
        """

        self._validate_inputs(target, action)

        for scope in self._policies:
            if scope.matches_target(target) and scope.permits(action):
                return True

        # Find the most specific scope (if any) for richer diagnostics.
        nearest: PolicyScope | None = None
        for scope in self._policies:
            if scope.matches_target(target):
                nearest = scope
                break

        reason = "no matching policy" if nearest is None else "action not in allowed_actions"
        _LOGGER.warning(
            "policy_gate_denied target=%r action=%r reason=%s",
            target,
            action,
            reason,
        )
        raise PolicyViolation(scope=nearest, action=action, reason=reason)

    def is_authorized(self, target: str, action: str) -> bool:
        """Boolean form of :meth:`check`; never raises for denial."""

        try:
            return self.check(target, action)
        except PolicyViolation:
            return False


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------


POLICY_GATE_REGISTRY: dict[str, PolicyGate] = {}

DEFAULT_POLICY_GATE: PolicyGate = PolicyGate()
POLICY_GATE_REGISTRY["default"] = DEFAULT_POLICY_GATE
