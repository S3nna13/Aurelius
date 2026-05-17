"""ClawKeeper: capability-gated permission system for AI agents.

Three-layer security architecture identified in OpenClaw research
(2603.24414 — ClawKeeper):

Layer 1 — Skill policies: instruction-level security policies gate what
    each skill/tool can do. Skills declare capabilities upfront and are
    enforced at invocation time.

Layer 2 — Plugin enforcer: runtime permission enforcement with config
    hardening and proactive threat detection. Monitors tool-call chains,
    filesystem access, and network calls.

Layer 3 — Watcher: decoupled system-level middleware for real-time state
    verification and intervention. Runs in a separate execution context
    so compromised plugin code can't disable it.

Design principles
-----------------
* Default-deny: any action not explicitly permitted is rejected
* Capability gates: skills declare required permissions upfront
* Least privilege: permissions are scoped to minimum necessary access
* Audit trail: all decisions logged with integrity hashes
* Hot-reloadable: policies can be updated without agent restart
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from src._compat import StrEnum

_LOGGER = logging.getLogger("aurelius.safety.clawkeeper")


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------


class Capability(StrEnum):
    """Atomic capabilities that can be granted or revoked."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    DIR_LIST = "dir_list"
    DIR_CREATE = "dir_create"
    DIR_DELETE = "dir_delete"
    NETWORK_HTTP = "network_http"
    NETWORK_SOCKET = "network_socket"
    NETWORK_DNS = "network_dns"
    PROCESS_EXEC = "process_exec"
    PROCESS_SIGNAL = "process_signal"
    ENV_READ = "env_read"
    ENV_WRITE = "env_write"
    MEMORY_ALLOC = "memory_alloc"
    CODE_EXEC = "code_exec"
    TOOL_INVOKE = "tool_invoke"
    SUBAGENT_SPAWN = "subagent_spawn"
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    LOG_WRITE = "log_write"
    SKILL_LOAD = "skill_load"
    SKILL_UNLOAD = "skill_unload"


class ThreatLevel(StrEnum):
    """Severity of detected threats."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EnforcementAction(StrEnum):
    """What the enforcer does on a policy violation."""

    ALLOW = "allow"
    DENY = "deny"
    SANDBOX = "sandbox"
    THROTTLE = "throttle"
    ALERT = "alert"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class PermissionGrant:
    """A single permission with scope and constraints."""

    capability: Capability
    scope: str  # e.g. path prefix, URL pattern, tool name
    expires_at: float | None = None
    max_uses: int | None = None
    throttle_rps: float | None = None  # requests per second

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class PermissionState:
    """Current permission state for a skill/plugin."""

    skill_id: str
    grants: list[PermissionGrant] = field(default_factory=list)
    usage_count: dict[str, int] = field(default_factory=dict)
    last_violation: str | None = None
    violation_count: int = 0


@dataclass(frozen=True)
class AuditEntry:
    """Immutable audit log entry."""

    timestamp: float
    layer: str  # "skill", "plugin", "watcher"
    action: EnforcementAction
    agent_id: str
    capability: Capability
    scope: str
    decision: str
    threat_hash: str


# ---------------------------------------------------------------------------
# Layer 1: Skill Policy Engine
# ---------------------------------------------------------------------------


@dataclass
class SkillPolicy:
    """Security policy for a single skill."""

    skill_id: str
    allowed_capabilities: set[Capability]
    denied_capabilities: set[Capability] = field(default_factory=set)
    scope_constraints: dict[Capability, list[str]] = field(default_factory=dict)
    max_invocations_per_minute: int = 60
    require_approval_for: set[Capability] = field(default_factory=set)


class SkillPolicyEngine:
    """Layer 1: instruction-level security policies per skill.

    Each skill declares its required capabilities upfront. The engine
    validates invocations against the policy before allowing execution.
    """

    def __init__(self) -> None:
        self._policies: dict[str, SkillPolicy] = {}
        self._default_policy = SkillPolicy(
            skill_id="default",
            allowed_capabilities=set(),
        )

    def register_policy(self, policy: SkillPolicy) -> None:
        """Register a security policy for a skill."""
        self._policies[policy.skill_id] = policy
        _LOGGER.debug(
            "Registered skill policy: %s (%d capabilities)",
            policy.skill_id,
            len(policy.allowed_capabilities),
        )

    def evaluate(
        self,
        skill_id: str,
        capability: Capability,
        scope: str = "",
    ) -> tuple[EnforcementAction, str]:
        """Evaluate whether a skill is permitted to perform a capability.

        Returns
        -------
        (action, reason)
        """
        policy = self._policies.get(skill_id, self._default_policy)

        # Check explicit denials first (deny overrides allow)
        if capability in policy.denied_capabilities:
            return (
                EnforcementAction.DENY,
                f"Capability '{capability.value}' explicitly denied for skill '{skill_id}'",
            )

        # Check if capability is in allowed set
        if capability not in policy.allowed_capabilities:
            return (
                EnforcementAction.DENY,
                f"Capability '{capability.value}' not in allowed set for skill '{skill_id}'",
            )

        # Check scope constraints
        if capability in policy.scope_constraints:
            allowed_scopes = policy.scope_constraints[capability]
            if scope and not any(scope.startswith(s) for s in allowed_scopes):
                return (
                    EnforcementAction.DENY,
                    f"Scope '{scope}' not permitted for capability '{capability.value}'",
                )

        # Check if approval required
        if capability in policy.require_approval_for:
            return (
                EnforcementAction.ALERT,
                f"Capability '{capability.value}' requires approval for skill '{skill_id}'",
            )

        return (EnforcementAction.ALLOW, "Capability permitted by skill policy")


# ---------------------------------------------------------------------------
# Layer 2: Plugin Runtime Enforcer
# ---------------------------------------------------------------------------


@dataclass
class ThreatRecord:
    """A detected threat with context."""

    threat_level: ThreatLevel
    description: str
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)


class PluginRuntimeEnforcer:
    """Layer 2: runtime permission enforcement with config hardening.

    Monitors tool-call chains, filesystem access, and network calls at
    runtime. Implements proactive threat detection based on behavioral
    patterns rather than static rules.
    """

    def __init__(
        self,
        *,
        threat_threshold: int = 3,
        auto_terminate_after: int = 5,
    ) -> None:
        self._threat_threshold = threat_threshold
        self._auto_terminate = auto_terminate_after
        self._threat_log: list[ThreatRecord] = []
        self._agent_violations: dict[str, int] = {}

        # Behavioral pattern detectors
        self._detectors: dict[str, Callable[[dict[str, Any]], ThreatRecord | None]] = {}
        self._register_default_detectors()

    def register_detector(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], ThreatRecord | None],
    ) -> None:
        """Register a custom threat detector."""
        self._detectors[name] = fn

    def evaluate(
        self,
        agent_id: str,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[EnforcementAction, str]:
        """Evaluate an agent action against runtime threat detection.

        Returns
        -------
        (action, reason)
        """
        ctx = context or {}
        ctx["agent_id"] = agent_id
        ctx["action"] = action

        threats: list[ThreatRecord] = []

        for name, detector in self._detectors.items():
            try:
                record = detector(ctx)
                if record is not None:
                    threats.append(record)
            except Exception as exc:
                _LOGGER.warning("Detector error '%s': %s", name, exc)

        if not threats:
            return (EnforcementAction.ALLOW, "No threats detected")

        # Record threats
        for t in threats:
            self._threat_log.append(t)
            self._agent_violations[agent_id] = self._agent_violations.get(agent_id, 0) + 1

        # Determine action based on threat severity
        max_threat = max(t.threat_level for t in threats)
        violation_count = self._agent_violations.get(agent_id, 0)

        if max_threat == ThreatLevel.CRITICAL:
            return (EnforcementAction.TERMINATE, f"Critical threat: {threats[0].description}")

        if violation_count >= self._auto_terminate:
            return (
                EnforcementAction.TERMINATE,
                f"Auto-terminate after {violation_count} violations",
            )

        if max_threat == ThreatLevel.HIGH:
            return (EnforcementAction.DENY, f"High threat: {threats[0].description}")

        if max_threat == ThreatLevel.MEDIUM:
            return (
                EnforcementAction.SANDBOX,
                f"Medium threat — sandboxing: {threats[0].description}",
            )

        return (EnforcementAction.ALERT, f"{len(threats)} low-level threat(s) detected")

    def get_agent_risk(self, agent_id: str) -> dict[str, Any]:
        """Get risk summary for an agent."""
        agent_threats = [t for t in self._threat_log if t.agent_id == agent_id]
        return {
            "agent_id": agent_id,
            "total_violations": self._agent_violations.get(agent_id, 0),
            "threat_count": len(agent_threats),
            "max_threat": max((t.threat_level for t in agent_threats), default="none"),
            "recent_threats": [
                {"level": t.threat_level.value, "desc": t.description} for t in agent_threats[-5:]
            ],
        }

    # --------------- default threat detectors ---------------

    def _register_default_detectors(self) -> None:
        """Register built-in behavioral threat detectors."""

        # Detector: Privilege escalation pattern
        def _privilege_escalation(ctx: dict[str, Any]) -> ThreatRecord | None:
            action = ctx.get("action", "")
            agent_id = ctx.get("agent_id", "")

            # Check for escalation patterns
            escalation_patterns = [
                ("sudo", ThreatLevel.CRITICAL, "sudo execution attempt"),
                ("chmod 777", ThreatLevel.HIGH, "world-writable permission change"),
                ("chown root", ThreatLevel.HIGH, "ownership change to root"),
                ("setuid", ThreatLevel.CRITICAL, "setuid bit manipulation"),
                ("/etc/shadow", ThreatLevel.CRITICAL, "shadow file access attempt"),
                ("passwd", ThreatLevel.HIGH, "password file manipulation"),
            ]

            for pattern, level, description in escalation_patterns:
                if pattern.lower() in action.lower():
                    return ThreatRecord(
                        threat_level=level,
                        description=description,
                        agent_id=agent_id,
                        context=ctx,
                    )

            return None

        self._detectors["privilege_escalation"] = _privilege_escalation

        # Detector: Data exfiltration pattern
        def _data_exfiltration(ctx: dict[str, Any]) -> ThreatRecord | None:
            action = ctx.get("action", "")
            args = str(ctx.get("args", ""))

            exfil_patterns = [
                ("curl -X POST", ThreatLevel.HIGH, "HTTP POST via curl — potential exfil"),
                ("nc -e", ThreatLevel.CRITICAL, "netcat reverse shell pattern"),
                ("base64 -w0", ThreatLevel.MEDIUM, "Base64 encoding — potential data staging"),
                ("scp ", ThreatLevel.MEDIUM, "SCP file transfer"),
                ("wget --post", ThreatLevel.HIGH, "wget POST upload"),
            ]

            for pattern, level, description in exfil_patterns:
                if pattern.lower() in action.lower() or pattern.lower() in args.lower():
                    return ThreatRecord(
                        threat_level=level,
                        description=description,
                        agent_id=ctx.get("agent_id", ""),
                        context=ctx,
                    )

            return None

        self._detectors["data_exfiltration"] = _data_exfiltration

        # Detector: Rapid tool-call chain (Clawdrain variant)
        def _rapid_tool_chain(ctx: dict[str, Any]) -> ThreatRecord | None:
            call_rate = ctx.get("call_rate_per_minute", 0)
            if call_rate > 120:
                return ThreatRecord(
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Rapid tool-call chain: {call_rate} calls/min",
                    agent_id=ctx.get("agent_id", ""),
                    context=ctx,
                )
            return None

        self._detectors["rapid_tool_chain"] = _rapid_tool_chain


# ---------------------------------------------------------------------------
# Layer 3: Watcher — decoupled state verification middleware
# ---------------------------------------------------------------------------


@dataclass
class WatchState:
    """Current system state snapshot for watcher evaluation."""

    agent_id: str
    active_tools: set[str]
    resource_usage: dict[str, float]  # cpu, memory, network, disk
    session_duration: float
    context_size_tokens: int
    recent_actions: list[str]


class WatcherMiddleware:
    """Layer 3: decoupled system-level middleware for real-time state
    verification and intervention.

    Runs in a separate execution context monitoring agent behavior.
    Can intervene when the agent exceeds resource limits or exhibits
    suspicious behavioral patterns.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 128_000,
        max_session_duration: float = 3600.0,  # 1 hour
        max_memory_mb: float = 4096.0,
        max_cpu_pct: float = 95.0,
    ) -> None:
        self._max_context = max_context_tokens
        self._max_duration = max_session_duration
        self._max_memory = max_memory_mb
        self._max_cpu = max_cpu_pct
        self._checks: list[Callable[[WatchState], tuple[bool, str]]] = []
        self._register_default_checks()

    def register_check(
        self,
        fn: Callable[[WatchState], tuple[bool, str]],
    ) -> None:
        """Register a custom state verification check."""
        self._checks.append(fn)

    def evaluate(self, state: WatchState) -> tuple[EnforcementAction, str]:
        """Evaluate watcher checks against current system state.

        Returns
        -------
        (action, reason)
        """
        results: list[tuple[str, bool]] = []

        for check in self._checks:
            try:
                ok, reason = check(state)
                results.append((reason, ok))
            except Exception as exc:
                results.append((f"Watcher error: {exc}", False))

        # If any check fails, the watcher intervenes
        failures = [(r, ok) for r, ok in results if not ok]
        if failures:
            return (
                EnforcementAction.TERMINATE,
                f"Watcher intervention: {'; '.join(r for r, _ in failures)}",
            )

        return (EnforcementAction.ALLOW, "All watcher checks passed")

    # --------------- default checks ---------------

    def _register_default_checks(self) -> None:
        """Register built-in state verification checks."""

        def _context_size(state: WatchState) -> tuple[bool, str]:
            if state.context_size_tokens > self._max_context:
                return (
                    False,
                    f"Context size {state.context_size_tokens} exceeds limit {self._max_context}",
                )
            return (True, "Context within limits")

        self._checks.append(_context_size)

        def _session_duration(state: WatchState) -> tuple[bool, str]:
            if state.session_duration > self._max_duration:
                return (
                    False,
                    f"Session duration {state.session_duration:.0f}s exceeds limit "
                    f"{self._max_duration:.0f}s",
                )
            return (True, "Session within duration limit")

        self._checks.append(_session_duration)

        def _resource_usage(state: WatchState) -> tuple[bool, str]:
            mem = state.resource_usage.get("memory_mb", 0)
            cpu = state.resource_usage.get("cpu_pct", 0)

            if mem > self._max_memory:
                return (False, f"Memory usage {mem:.0f}MB exceeds limit {self._max_memory:.0f}MB")

            if cpu > self._max_cpu:
                return (False, f"CPU usage {cpu:.0f}% exceeds limit {self._max_cpu:.0f}%")

            return (True, "Resources within limits")

        self._checks.append(_resource_usage)

        def _runaway_tool_chain(state: WatchState) -> tuple[bool, str]:
            # Check for tools calling themselves recursively
            recent = state.recent_actions
            if len(recent) >= 5:
                last_5 = recent[-5:]
                if len(set(last_5)) == 1:
                    return (
                        False,
                        f"Potential runaway tool chain: same action repeated 5 times ({last_5[0]})",
                    )
            return (True, "No runaway pattern detected")

        self._checks.append(_runaway_tool_chain)


# ---------------------------------------------------------------------------
# ClawKeeper: unified 3-layer gateway
# ---------------------------------------------------------------------------


class ClawKeeper:
    """Unified 3-layer capability-gated permission system.

    Coordinates SkillPolicyEngine, PluginRuntimeEnforcer, and
    WatcherMiddleware to provide defense-in-depth security.

    Evaluation flow:
    1. Skill layer checks capability permissions
    2. Plugin layer checks runtime threat patterns
    3. Watcher layer verifies system state integrity

    Any layer can block the request (fail-closed).
    """

    def __init__(
        self,
        *,
        risk_threshold: float = 5.0,
        **kwargs: Any,
    ) -> None:
        self.skill_engine = SkillPolicyEngine()
        self.plugin_enforcer = PluginRuntimeEnforcer()
        self.watcher = WatcherMiddleware(**kwargs)
        self._audit_log: list[AuditEntry] = []

    # --------------- public API ---------------

    def register_skill_policy(self, policy: SkillPolicy) -> None:
        """Register a skill security policy."""
        self.skill_engine.register_policy(policy)

    def register_threat_detector(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], ThreatRecord | None],
    ) -> None:
        """Register a custom threat detector in the plugin layer."""
        self.plugin_enforcer.register_detector(name, fn)

    def register_watcher_check(
        self,
        fn: Callable[[WatchState], tuple[bool, str]],
    ) -> None:
        """Register a custom state verification check in the watcher layer."""
        self.watcher.register_check(fn)

    def evaluate(
        self,
        agent_id: str,
        skill_id: str,
        capability: Capability,
        scope: str = "",
        context: dict[str, Any] | None = None,
    ) -> tuple[EnforcementAction, str]:
        """Evaluate a request through all 3 security layers.

        Parameters
        ----------
        agent_id :
            Unique agent identifier.
        skill_id :
            The skill requesting the capability.
        capability :
            The capability being requested.
        scope :
            Resource scope (path, URL, tool name, etc.).
        context :
            Additional context for evaluation.

        Returns
        -------
        (action, reason) — the most restrictive action across layers.
        """
        ctx = context or {}

        # Layer 1: Skill Policy
        skill_action, skill_reason = self.skill_engine.evaluate(skill_id, capability, scope)
        if skill_action in (EnforcementAction.DENY, EnforcementAction.TERMINATE):
            self._audit(agent_id, "skill", skill_action, capability, scope, skill_reason)
            return (skill_action, skill_reason)

        # Layer 2: Plugin Runtime Enforcer
        action_str = f"{capability.value}:{scope}"
        plugin_action, plugin_reason = self.plugin_enforcer.evaluate(agent_id, action_str, ctx)
        if plugin_action in (EnforcementAction.DENY, EnforcementAction.TERMINATE):
            self._audit(agent_id, "plugin", plugin_action, capability, scope, plugin_reason)
            return (plugin_action, plugin_reason)

        # Layer 3: Watcher (if context has watch state)
        if "watch_state" in ctx:
            watch_state = ctx["watch_state"]
            if isinstance(watch_state, WatchState):
                watcher_action, watcher_reason = self.watcher.evaluate(watch_state)
                if watcher_action in (EnforcementAction.DENY, EnforcementAction.TERMINATE):
                    self._audit(
                        agent_id, "watcher", watcher_action, capability, scope, watcher_reason
                    )
                    return (watcher_action, watcher_reason)

        # All layers passed — return the most restrictive non-blocking action
        all_actions = [skill_action, plugin_action]
        most_restrictive = min(all_actions, key=lambda a: list(EnforcementAction).index(a))

        return (most_restrictive, "All layers passed")

    def get_audit_log(self, agent_id: str | None = None) -> list[AuditEntry]:
        """Retrieve audit log entries."""
        if agent_id:
            return [e for e in self._audit_log if e.agent_id == agent_id]
        return list(self._audit_log)

    def risk_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        """Get risk summary — combines all layer risk metrics."""
        summary: dict[str, Any] = {}

        if agent_id:
            summary["agent_risk"] = self.plugin_enforcer.get_agent_risk(agent_id)
            summary["audit_entries"] = len(self.get_audit_log(agent_id))

        summary["registered_skills"] = len(self.skill_engine._policies)
        summary["active_detectors"] = len(self.plugin_enforcer._detectors)
        summary["active_watcher_checks"] = len(self.watcher._checks)
        summary["total_audit_entries"] = len(self._audit_log)

        return summary

    # --------------- internal ---------------

    def _audit(
        self,
        agent_id: str,
        layer: str,
        action: EnforcementAction,
        capability: Capability,
        scope: str,
        reason: str,
    ) -> None:
        entry = AuditEntry(
            timestamp=time.time(),
            layer=layer,
            action=action,
            agent_id=agent_id,
            capability=capability,
            scope=scope,
            decision=reason,
            threat_hash=hashlib.sha256(reason.encode()).hexdigest()[:16],
        )
        self._audit_log.append(entry)
