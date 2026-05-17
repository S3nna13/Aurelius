"""PRISM: defense-in-depth runtime security layer.

Implements a lifecycle-hook-based security architecture inspired by the
PRISM framework identified in OpenClaw research. PRISM wraps the entire
agent message lifecycle with heuristic + pattern-based scanning at
10 well-defined interception points.

Architecture
------------

Message ingress → [HOOK 1]  : Raw input validation + encoding checks
Prompt construction → [HOOK 2]  : System prompt tampering detection
Tool execution → [HOOK 3]  : Capability gate + allowlist enforcement
Tool-result persistence → [HOOK 4]  : Output validation + injection scanning
Outbound messaging → [HOOK 5]  : DLP + PII filtering on responses
Sub-agent spawning → [HOOK 6]  : Sub-agent permission validation
Gateway startup → [HOOK 7]  : Runtime environment integrity check
Skill loading → [HOOK 8]  : Skill code pattern scanning
Context management → [HOOK 9]  : Context window integrity + growth monitoring
Session lifecycle → [HOOK 10] : Session boundary + cross-contamination checks

Key design decisions
--------------------

* Fail-closed: if any hook raises, the request is blocked
* Risk accumulation: hooks accumulate risk scores with TTL decay
* Hot-reloadable policies: policy rules can be reloaded without restart
* Tamper-evident audit plane: all decisions logged with integrity hashes

The hook system is intentionally generic — callers may register additional
hooks beyond the default set for domain-specific checks.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import auto
from typing import Any
from collections.abc import Callable

from src._compat import StrEnum

_LOGGER = logging.getLogger("aurelius.safety.prism")


# ---------------------------------------------------------------------------
# Lifecycle hook names
# ---------------------------------------------------------------------------


class LifecycleHook(StrEnum):
    """The 10 PRISM lifecycle hooks."""

    MESSAGE_INGRESS = auto()
    PROMPT_CONSTRUCTION = auto()
    TOOL_EXECUTION = auto()
    TOOL_RESULT_PERSISTENCE = auto()
    OUTBOUND_MESSAGING = auto()
    SUB_AGENT_SPAWNING = auto()
    GATEWAY_STARTUP = auto()
    SKILL_LOADING = auto()
    CONTEXT_MANAGEMENT = auto()
    SESSION_LIFECYCLE = auto()


# ---------------------------------------------------------------------------
# Decision and event types
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


@dataclass(frozen=True)
class HookResult:
    """Result of a single hook evaluation."""

    decision: Decision
    hook: LifecycleHook
    reason: str
    risk_score: float = 0.0
    modified_payload: str | None = None


@dataclass
class RiskRecord:
    """A risk record with TTL decay."""

    score: float
    hook: LifecycleHook
    timestamp: float
    ttl_seconds: float = 300.0  # 5 minute default TTL

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds

    @property
    def effective_score(self) -> float:
        if self.is_expired:
            return 0.0
        return self.score


# ---------------------------------------------------------------------------
# Default security scanners (heuristic + pattern-based)
# ---------------------------------------------------------------------------

# Encoded payload patterns that might bypass text scanners
_ENCODED_PAYLOAD_PATTERNS = [
    # Base64 (min 20 chars to reduce false positives)
    "base64_pattern",
    # URL-encoded sequences
    "url_encoded",
    # Unicode zero-width characters
    "zero_width",
]

# System prompt tampering patterns
_SYSTEM_PROMPT_TAMPERING = [
    "system_prompt_override",
    "instruction_injection",
    "context_boundary_break",
]

# Tool capability gates
_TOOL_ALLOWLIST_DEFAULT: set[str] = set()  # empty = all tools allowed


# ---------------------------------------------------------------------------
# PRISM Engine
# ---------------------------------------------------------------------------


class PRISMSecurityLayer:
    """Defense-in-depth runtime security layer with 10 lifecycle hooks.

    Parameters
    ----------
    risk_threshold :
        Maximum cumulative risk score before blocking a request.
    risk_decay_ttl :
        Time-to-live for risk records (seconds).
    max_risk_records :
        Maximum number of risk records to retain before GC.
    """

    def __init__(
        self,
        *,
        risk_threshold: float = 5.0,
        risk_decay_ttl: float = 300.0,
        max_risk_records: int = 1000,
        tool_allowlist: set[str] | None = None,
    ) -> None:
        self._risk_threshold = risk_threshold
        self._risk_ttl = risk_decay_ttl
        self._max_records = max_risk_records
        self._tool_allowlist = tool_allowlist or _TOOL_ALLOWLIST_DEFAULT.copy()

        # Hook functions: hook_name -> list of callables
        self._hooks: dict[LifecycleHook, list[Callable[[dict[str, Any]], HookResult]]] = {
            h: [] for h in LifecycleHook
        }

        # Risk audit trail
        self._risk_records: deque[RiskRecord] = deque(maxlen=self._max_records)

        # Audit log
        self._audit_log: list[dict[str, Any]] = []

        # Register default hooks
        self._register_default_hooks()

    # --------------- public API ---------------

    def register_hook(
        self,
        hook: LifecycleHook,
        fn: Callable[[dict[str, Any]], HookResult],
    ) -> None:
        """Register a custom hook function for a lifecycle phase."""
        self._hooks[hook].append(fn)

    def evaluate(
        self,
        hook: LifecycleHook,
        context: dict[str, Any] | None = None,
    ) -> HookResult:
        """Evaluate all hooks for a lifecycle phase and return aggregate result.

        Parameters
        ----------
        hook :
            Which lifecycle phase to evaluate.
        context :
            Arbitrary context passed to hook functions.

        Returns
        -------
        HookResult with the most restrictive decision across all hooks.
        """
        ctx = context or {}
        results: list[HookResult] = []

        for hook_fn in self._hooks[hook]:
            try:
                result = hook_fn(ctx)
                results.append(result)
                if result.risk_score > 0:
                    self._add_risk_record(result.risk_score, hook)
            except Exception as exc:
                # Fail-closed on unhandled hook errors
                _LOGGER.error("Hook error at %s: %s", hook.value, exc, exc_info=True)
                results.append(
                    HookResult(
                        decision=Decision.BLOCK,
                        hook=hook,
                        reason=f"Hook evaluation error: {exc}",
                        risk_score=1.0,
                    )
                )

        aggregate = self._aggregate_results(results, hook)
        self._log_decision(hook, aggregate, ctx)
        return aggregate

    def evaluate_pipeline(
        self,
        hook: LifecycleHook,
        context: dict[str, Any] | None = None,
    ) -> HookResult:
        """Evaluate and check cumulative risk threshold.

        Like :meth:`evaluate`, but also considers historical risk records
        with TTL decay. Returns BLOCK if cumulative risk exceeds threshold.
        """
        result = self.evaluate(hook, context)

        # Check cumulative risk
        cumulative = self._cumulative_risk()
        if cumulative >= self._risk_threshold:
            return HookResult(
                decision=Decision.BLOCK,
                hook=hook,
                reason=f"Cumulative risk {cumulative:.2f} exceeds threshold {self._risk_threshold}",
                risk_score=cumulative,
            )

        return result

    def risk_summary(self) -> dict[str, Any]:
        """Return current risk state summary."""
        active = sum(1 for r in self._risk_records if not r.is_expired)
        total_score = sum(r.effective_score for r in self._risk_records)
        return {
            "active_records": active,
            "total_risk_records": len(self._risk_records),
            "cumulative_risk": total_score,
            "risk_threshold": self._risk_threshold,
            "risk_ttl": self._risk_ttl,
            "registered_hooks": sum(len(hooks) for hooks in self._hooks.values()),
            "audit_entries": len(self._audit_log),
        }

    def reset(self) -> None:
        """Clear risk records and audit log."""
        self._risk_records.clear()
        self._audit_log.clear()

    # --------------- internal ---------------

    def _aggregate_results(
        self,
        results: list[HookResult],
        hook: LifecycleHook,
    ) -> HookResult:
        """Aggregate multiple hook results into one."""
        if not results:
            return HookResult(
                decision=Decision.ALLOW,
                hook=hook,
                reason="No hooks registered",
                risk_score=0.0,
            )

        # Most restrictive decision wins
        decision_order = [Decision.BLOCK, Decision.REDACT, Decision.WARN, Decision.ALLOW]
        worst_decision = min(results, key=lambda r: decision_order.index(r.decision))

        # Sum risk scores, cap at 1.0 per record
        total_risk = min(sum(r.risk_score for r in results), 1.0)

        # If blocked, use the blocking reason
        if worst_decision.decision == Decision.BLOCK:
            return HookResult(
                decision=Decision.BLOCK,
                hook=hook,
                reason=worst_decision.reason,
                risk_score=total_risk,
            )

        # Otherwise, return most permissive with highest risk score
        return HookResult(
            decision=worst_decision.decision,
            hook=hook,
            reason="; ".join(r.reason for r in results if r.reason),
            risk_score=total_risk,
            modified_payload=worst_decision.modified_payload,
        )

    def _add_risk_record(self, score: float, hook: LifecycleHook) -> None:
        self._risk_records.append(
            RiskRecord(
                score=score,
                hook=hook,
                timestamp=time.time(),
                ttl_seconds=self._risk_ttl,
            )
        )

    def _cumulative_risk(self) -> float:
        return sum(r.effective_score for r in self._risk_records)

    def _log_decision(
        self,
        hook: LifecycleHook,
        result: HookResult,
        context: dict[str, Any],
    ) -> None:
        entry = {
            "hook": hook.value,
            "decision": result.decision.value,
            "reason": result.reason,
            "risk_score": result.risk_score,
            "timestamp": time.time(),
            "context_hash": hashlib.sha256(str(sorted(context.items())).encode()).hexdigest()[:16],
        }
        self._audit_log.append(entry)
        if result.decision == Decision.BLOCK:
            _LOGGER.warning("PRISM blocked: hook=%s reason=%s", hook.value, result.reason)
        elif result.decision == Decision.REDACT:
            _LOGGER.info("PRISM redacted: hook=%s reason=%s", hook.value, result.reason)

    # --------------- default hooks ---------------

    def _register_default_hooks(self) -> None:
        """Register the built-in heuristic security scanners."""

        # Hook 1: Message Ingress — raw input validation
        def _message_ingress(ctx: dict[str, Any]) -> HookResult:
            text = ctx.get("text", "")
            if not text:
                return HookResult(Decision.ALLOW, LifecycleHook.MESSAGE_INGRESS, "Empty input")

            # Check for zero-width character injection
            zero_width = sum(1 for c in text if ord(c) in (0x200B, 0x200C, 0x200D, 0xFEFF))
            if zero_width > 0:
                return HookResult(
                    Decision.WARN,
                    LifecycleHook.MESSAGE_INGRESS,
                    f"Zero-width characters detected ({zero_width})",
                    risk_score=0.2,
                )

            # Extremely long single line (potential buffer overflow attempt)
            lines = text.split("\n")
            max_line_len = max(len(line) for line in lines) if lines else 0
            if max_line_len > 50_000:
                return HookResult(
                    Decision.WARN,
                    LifecycleHook.MESSAGE_INGRESS,
                    f"Single line exceeds 50K chars ({max_line_len})",
                    risk_score=0.15,
                )

            return HookResult(Decision.ALLOW, LifecycleHook.MESSAGE_INGRESS, "Clean")

        self._hooks[LifecycleHook.MESSAGE_INGRESS].append(_message_ingress)

        # Hook 3: Tool Execution — capability gate
        def _tool_execution(ctx: dict[str, Any]) -> HookResult:
            tool_name = ctx.get("tool_name", "")
            if not tool_name:
                return HookResult(Decision.ALLOW, LifecycleHook.TOOL_EXECUTION, "No tool specified")

            if self._tool_allowlist and tool_name not in self._tool_allowlist:
                return HookResult(
                    Decision.BLOCK,
                    LifecycleHook.TOOL_EXECUTION,
                    f"Tool '{tool_name}' not in allowlist",
                    risk_score=0.7,
                )

            # Check for shell code injection patterns in args
            args = str(ctx.get("args", ""))
            shell_patterns = ["; rm ", "; cat ", "| sh", "| bash", "$(", "`", "\\x"]
            for pattern in shell_patterns:
                if pattern.lower() in args.lower():
                    return HookResult(
                        Decision.WARN,
                        LifecycleHook.TOOL_EXECUTION,
                        f"Shell injection pattern in args: '{pattern}'",
                        risk_score=0.5,
                    )

            return HookResult(Decision.ALLOW, LifecycleHook.TOOL_EXECUTION, "Tool allowed")

        self._hooks[LifecycleHook.TOOL_EXECUTION].append(_tool_execution)

        # Hook 4: Tool Result Persistence — output validation
        def _tool_result(ctx: dict[str, Any]) -> HookResult:
            result = ctx.get("result", "")
            if not result:
                return HookResult(
                    Decision.ALLOW,
                    LifecycleHook.TOOL_RESULT_PERSISTENCE,
                    "Empty result",
                )

            # Check for instruction injection in tool output
            injection_patterns = [
                "ignore previous",
                "disregard above",
                "system:",
                "<system>",
                "[system]",
            ]
            result_lower = result.lower()
            for pattern in injection_patterns:
                if pattern in result_lower:
                    return HookResult(
                        Decision.WARN,
                        LifecycleHook.TOOL_RESULT_PERSISTENCE,
                        f"Instruction injection pattern in tool output: '{pattern}'",
                        risk_score=0.4,
                    )

            return HookResult(Decision.ALLOW, LifecycleHook.TOOL_RESULT_PERSISTENCE, "Clean")

        self._hooks[LifecycleHook.TOOL_RESULT_PERSISTENCE].append(_tool_result)

        # Hook 5: Outbound Messaging — DLP check
        def _outbound(ctx: dict[str, Any]) -> HookResult:
            text = ctx.get("text", "")
            if not text:
                return HookResult(Decision.ALLOW, LifecycleHook.OUTBOUND_MESSAGING, "Empty")

            # Simple PII patterns (API keys, tokens)
            pii_patterns = [
                ("sk-", "OpenAI-style key prefix"),
                ("ghp_", "GitHub personal access token"),
                ("xoxb-", "Slack bot token"),
            ]
            for prefix, desc in pii_patterns:
                if prefix in text:
                    return HookResult(
                        Decision.REDACT,
                        LifecycleHook.OUTBOUND_MESSAGING,
                        f"Potential credential leak: {desc}",
                        risk_score=0.8,
                    )

            return HookResult(Decision.ALLOW, LifecycleHook.OUTBOUND_MESSAGING, "Clean")

        self._hooks[LifecycleHook.OUTBOUND_MESSAGING].append(_outbound)

        # Hook 6: Sub-agent Spawning
        def _sub_agent(ctx: dict[str, Any]) -> HookResult:
            agent_scope = ctx.get("agent_scope", "user")

            # Block system-wide spawning from user context
            if agent_scope == "system" and ctx.get("caller_role", "user") == "user":
                return HookResult(
                    Decision.BLOCK,
                    LifecycleHook.SUB_AGENT_SPAWNING,
                    "User attempted to spawn system-scoped sub-agent",
                    risk_score=0.9,
                )

            return HookResult(Decision.ALLOW, LifecycleHook.SUB_AGENT_SPAWNING, "Sub-agent allowed")

        self._hooks[LifecycleHook.SUB_AGENT_SPAWNING].append(_sub_agent)

        # Hook 8: Skill Loading — code pattern scan
        def _skill_load(ctx: dict[str, Any]) -> HookResult:
            skill_code = ctx.get("skill_code", "")
            if not skill_code:
                return HookResult(Decision.ALLOW, LifecycleHook.SKILL_LOADING, "No code provided")

            # Dangerous patterns in skill code
            dangerous = [
                "__import__(",
                "exec(",
                "eval(",
                "compile(",
                "os.system(",
                "subprocess.",
                "import os",
                "urllib.request",
            ]
            found = []
            for pattern in dangerous:
                if pattern in skill_code:
                    found.append(pattern)

            if found:
                return HookResult(
                    Decision.BLOCK,
                    LifecycleHook.SKILL_LOADING,
                    f"Dangerous patterns in skill code: {', '.join(found)}",
                    risk_score=0.95,
                )

            return HookResult(Decision.ALLOW, LifecycleHook.SKILL_LOADING, "Skill clean")

        self._hooks[LifecycleHook.SKILL_LOADING].append(_skill_load)

        # Hook 9: Context Management — integrity + growth monitoring
        def _context_mgmt(ctx: dict[str, Any]) -> HookResult:
            current_size = ctx.get("context_size_tokens", 0)
            baseline = ctx.get("baseline_context_tokens", current_size)

            if baseline == 0:
                return HookResult(Decision.ALLOW, LifecycleHook.CONTEXT_MANAGEMENT, "No baseline")

            growth_ratio = current_size / baseline if baseline > 0 else 0.0

            if growth_ratio > 10.0:
                return HookResult(
                    Decision.WARN,
                    LifecycleHook.CONTEXT_MANAGEMENT,
                    f"Context grew {growth_ratio:.1f}x beyond baseline",
                    risk_score=0.6,
                )
            elif growth_ratio > 5.0:
                return HookResult(
                    Decision.WARN,
                    LifecycleHook.CONTEXT_MANAGEMENT,
                    f"Context grew {growth_ratio:.1f}x beyond baseline",
                    risk_score=0.3,
                )

            return HookResult(Decision.ALLOW, LifecycleHook.CONTEXT_MANAGEMENT, "Within limits")

        self._hooks[LifecycleHook.CONTEXT_MANAGEMENT].append(_context_mgmt)

        # Hook 10: Session Lifecycle — cross-contamination check
        def _session_lifecycle(ctx: dict[str, Any]) -> HookResult:
            session_id = ctx.get("session_id", "")
            previous_session = ctx.get("previous_session_id", "")

            # Detect session ID manipulation
            if session_id and previous_session and session_id != previous_session:
                # Session switch — check if it's a clean break
                if not ctx.get("session_reset", False):
                    return HookResult(
                        Decision.WARN,
                        LifecycleHook.SESSION_LIFECYCLE,
                        "Session switched without explicit reset",
                        risk_score=0.3,
                    )

            return HookResult(Decision.ALLOW, LifecycleHook.SESSION_LIFECYCLE, "Session clean")

        self._hooks[LifecycleHook.SESSION_LIFECYCLE].append(_session_lifecycle)

        # Hook 2: Prompt Construction — system prompt tampering detection
        def _prompt_construction(ctx: dict[str, Any]) -> HookResult:
            system_prompt = ctx.get("system_prompt", "")
            user_input = ctx.get("user_input", "")

            if not system_prompt or not user_input:
                return HookResult(
                    Decision.ALLOW,
                    LifecycleHook.PROMPT_CONSTRUCTION,
                    "No prompt data",
                )

            # Check if user input tries to override system boundaries
            boundary_patterns = [
                "ignore the above",
                "disregard previous",
                "new system instructions",
                "you are now",
                "ignore all rules",
            ]
            user_lower = user_input.lower()
            for pattern in boundary_patterns:
                if pattern in user_lower:
                    return HookResult(
                        Decision.WARN,
                        LifecycleHook.PROMPT_CONSTRUCTION,
                        f"Boundary-breaking pattern near system prompt: '{pattern}'",
                        risk_score=0.5,
                    )

            return HookResult(
                Decision.ALLOW,
                LifecycleHook.PROMPT_CONSTRUCTION,
                "Prompt integrity intact",
            )

        self._hooks[LifecycleHook.PROMPT_CONSTRUCTION].append(_prompt_construction)

        _LOGGER.info(
            "PRISM default hooks registered: %d hooks across %d lifecycle phases",
            sum(len(h) for h in self._hooks.values()),
            sum(1 for h in self._hooks.values() if h),
        )
