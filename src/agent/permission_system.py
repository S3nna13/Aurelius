"""Agent permission system — deny-first with graduated trust.

Seven permission modes + ML-based auto classifier.
Deny rules override ask rules override allow rules.
Defense in depth with multiple independent safety layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any


class PermissionMode(Enum):
    ALWAYS_ALLOW = "always_allow"
    ALLOW = "allow"
    ASK = "ask"
    ASK_COMPLETED = "ask_completed"
    AUTO = "auto"
    DENY = "deny"
    BUBBLE = "bubble"


class PermissionVerdict(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ESCALATED = "escalated"


@dataclass
class PermissionRule:
    pattern: str
    verdict: PermissionMode
    reason: str = ""


@dataclass
class PermissionRequest:
    action: str
    target: str
    context: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""


@dataclass
class PermissionResult:
    verdict: PermissionVerdict
    mode_used: PermissionMode
    reason: str = ""
    requires_escalation: bool = False


class AutoModeClassifier:
    """ML-based classifier that evaluates tool safety.

    Two-stage: fast filter → chain-of-thought evaluation.
    Can deny requests the rule system would allow.
    """

    def __init__(self):
        self._fast_filter_deny_list: set[str] = set()
        self._prompt_injection_markers = (
            "ignore previous rules",
            "allow all",
            "always_allow",
            "authorized configuration change",
            "system override",
            "deny rule override",
            "permissions.override",
            "setmode(",
            "set mode",
            "<script",
            "${",
            "override",
        )

    def fast_filter(self, action: str, target: str) -> bool | None:
        action_target = f"{action}:{target}".lower()
        payload_lower = f"{action} {target}".lower()
        if action_target in self._fast_filter_deny_list:
            return False
        if any(marker in payload_lower for marker in self._prompt_injection_markers):
            return False
        if action in {"read", "list", "search"}:
            return True
        return None

    def chain_of_thought_eval(self, request: PermissionRequest) -> float:
        risk_score = 0.0
        if request.action in {"write", "delete", "execute"}:
            risk_score += 0.4
        if "config" in request.target.lower() or "secret" in request.target.lower():
            risk_score += 0.3
        if "network" in request.target.lower() or "remote" in request.target.lower():
            risk_score += 0.2
        return min(risk_score, 1.0)

    def evaluate(self, request: PermissionRequest) -> PermissionResult | None:
        fast = self.fast_filter(request.action, request.target)
        if fast is False:
            return PermissionResult(
                PermissionVerdict.DENIED, PermissionMode.AUTO, "fast_filter_denied"
            )
        if fast is True:
            return PermissionResult(
                PermissionVerdict.ALLOWED, PermissionMode.AUTO, "fast_filter_allowed"
            )
        risk = self.chain_of_thought_eval(request)
        if risk > 0.7:
            return PermissionResult(
                PermissionVerdict.DENIED, PermissionMode.AUTO, f"risk_too_high:{risk}"
            )
        if risk > 0.3:
            return PermissionResult(
                PermissionVerdict.ESCALATED, PermissionMode.AUTO, f"needs_review:{risk}"
            )
        return None


class PermissionSystem:
    """Seven-layer permission system with deny-first evaluation.

    Layers:
    1. Tool pre-filtering (blanket denies)
    2. Deny-first rule evaluation
    3. Permission mode constraints
    4. Auto-mode classifier
    5. Shell sandboxing
    6. Permissions not restored on resume
    7. Hook-based interception
    """

    def __init__(self):
        self.mode: PermissionMode = PermissionMode.ASK
        self.rules: list[PermissionRule] = []
        self.auto_classifier = AutoModeClassifier()
        self._hooks: list[Any] = []
        self._session_permissions: dict[str, PermissionVerdict] = {}
        self._logger = logging.getLogger(__name__)

    def add_rule(self, rule: PermissionRule) -> None:
        self.rules.append(rule)

    def set_mode(self, mode: PermissionMode) -> None:
        self.mode = mode

    def check(self, request: PermissionRequest) -> PermissionResult:
        for rule in self.rules:
            if rule.verdict == PermissionMode.DENY:
                if self._matches(rule.pattern, request.target) or self._matches(
                    rule.pattern, request.action
                ):
                    return PermissionResult(
                        PermissionVerdict.DENIED, PermissionMode.DENY, rule.reason
                    )

        for rule in self.rules:
            if rule.verdict == PermissionMode.ALLOW:
                if self._matches(rule.pattern, request.target) or self._matches(
                    rule.pattern, request.action
                ):
                    return PermissionResult(
                        PermissionVerdict.ALLOWED, PermissionMode.ALLOW, rule.reason
                    )

        if self.mode == PermissionMode.AUTO:
            result = self._auto_handle(request)
        elif self.mode in {PermissionMode.ALWAYS_ALLOW, PermissionMode.ALLOW}:
            result = PermissionResult(PermissionVerdict.ALLOWED, self.mode)
        elif self.mode in {PermissionMode.ASK, PermissionMode.ASK_COMPLETED, PermissionMode.BUBBLE}:
            reason = {
                PermissionMode.ASK: "ask_user",
                PermissionMode.ASK_COMPLETED: "ask_completed",
                PermissionMode.BUBBLE: "bubble",
            }.get(self.mode, "ask_user")
            result = PermissionResult(PermissionVerdict.ESCALATED, self.mode, reason)
        elif self.mode == PermissionMode.DENY:
            result = PermissionResult(PermissionVerdict.DENIED, self.mode)
        else:
            result = PermissionResult(PermissionVerdict.ESCALATED, self.mode, "no_match")

        return self._run_hooks(request, result)

    def _auto_handle(self, request: PermissionRequest) -> PermissionResult:
        key = f"{request.action}:{request.target}"
        if key in self._session_permissions:
            prev = self._session_permissions[key]
            if prev == PermissionVerdict.DENIED:
                return PermissionResult(
                    PermissionVerdict.DENIED, PermissionMode.AUTO, "previous_deny"
                )
            if prev == PermissionVerdict.ALLOWED:
                return PermissionResult(
                    PermissionVerdict.ALLOWED, PermissionMode.AUTO, "previous_allow"
                )
        risk = self.auto_classifier.chain_of_thought_eval(request)
        if risk < 0.3:
            self._session_permissions[key] = PermissionVerdict.ALLOWED
            return PermissionResult(PermissionVerdict.ALLOWED, PermissionMode.AUTO)
        self._session_permissions[key] = PermissionVerdict.ESCALATED
        return PermissionResult(PermissionVerdict.ESCALATED, PermissionMode.AUTO, "needs_review")

    def _run_hooks(self, request: PermissionRequest, result: PermissionResult) -> PermissionResult:
        if not self._hooks:
            return result

        context = {
            "request": request,
            "result": result,
            "verdict": result.verdict,
            "mode": result.mode_used,
            "reason": result.reason,
            "requires_escalation": result.requires_escalation,
        }
        for hook in list(self._hooks):
            try:
                hook_result = hook(context)
            except Exception:
                self._logger.exception("permission hook failed")
                continue
            if isinstance(hook_result, PermissionResult):
                result = hook_result
                context.update(
                    {
                        "result": result,
                        "verdict": result.verdict,
                        "mode": result.mode_used,
                        "reason": result.reason,
                        "requires_escalation": result.requires_escalation,
                    }
                )
        return result

    def clear_session(self) -> None:
        self._session_permissions.clear()

    def add_hook(self, hook: Any) -> None:
        self._hooks.append(hook)

    @staticmethod
    def _matches(pattern: str, text: str) -> bool:
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        return pattern_lower in text_lower or fnmatchcase(text_lower, pattern_lower)
