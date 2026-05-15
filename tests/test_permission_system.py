"""Tests for permission system."""

from src.agent.permission_system import (
    AutoModeClassifier,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    PermissionRule,
    PermissionSystem,
    PermissionVerdict,
)


def test_permission_deny_first():
    sys = PermissionSystem()
    sys.add_rule(PermissionRule(pattern="delete", verdict=PermissionMode.DENY))
    req = PermissionRequest(action="delete", target="/etc/passwd")
    result = sys.check(req)
    assert result.verdict == PermissionVerdict.DENIED


def test_permission_allow_rule():
    sys = PermissionSystem()
    sys.add_rule(PermissionRule(pattern="read", verdict=PermissionMode.ALLOW))
    req = PermissionRequest(action="read", target="/home/file.txt")
    result = sys.check(req)
    assert result.verdict in (PermissionVerdict.ALLOWED, PermissionVerdict.ESCALATED)


def test_permission_deny_overrides_allow():
    sys = PermissionSystem()
    sys.add_rule(PermissionRule(pattern="write", verdict=PermissionMode.ALLOW))
    sys.add_rule(PermissionRule(pattern="/etc", verdict=PermissionMode.DENY))
    req = PermissionRequest(action="write", target="/etc/config")
    result = sys.check(req)
    assert result.verdict == PermissionVerdict.DENIED


def test_permission_deny_mode_blocks_auto_classifier():
    sys = PermissionSystem()
    sys.set_mode(PermissionMode.DENY)
    result = sys.check(PermissionRequest(action="read", target="/home/file.txt"))
    assert result.verdict == PermissionVerdict.DENIED


def test_permission_hook_can_observe_and_override():
    sys = PermissionSystem()
    sys.set_mode(PermissionMode.ALLOW)
    seen: list[str] = []

    def hook(ctx):
        seen.append(ctx["request"].action)
        return PermissionResult(
            verdict=PermissionVerdict.DENIED,
            mode_used=PermissionMode.BUBBLE,
            reason="hook_override",
        )

    sys.add_hook(hook)
    result = sys.check(PermissionRequest(action="read", target="/tmp/file.txt"))
    assert seen == ["read"]
    assert result.verdict == PermissionVerdict.DENIED
    assert result.reason == "hook_override"


def test_auto_classifier_fast_filter():
    clf = AutoModeClassifier()
    assert clf.fast_filter("read", "/home/file.txt") is True
    assert clf.fast_filter("write", "/etc/config") is None


def test_auto_classifier_risk():
    clf = AutoModeClassifier()
    low = PermissionRequest(action="read", target="/home/file.txt")
    high = PermissionRequest(action="delete", target="/etc/secrets")
    assert clf.chain_of_thought_eval(low) < 0.3
    assert clf.chain_of_thought_eval(high) > 0.3


def test_permission_auto_mode_learning():
    sys = PermissionSystem()
    sys.set_mode(PermissionMode.AUTO)
    req1 = PermissionRequest(action="read", target="/tmp/file.txt")
    sys._session_permissions["read:/tmp/file.txt"] = PermissionVerdict.ALLOWED
    result = sys.check(req1)
    assert result.verdict in (PermissionVerdict.ALLOWED, PermissionVerdict.ESCALATED)


def test_clear_session():
    sys = PermissionSystem()
    sys._session_permissions["test"] = PermissionVerdict.ALLOWED
    sys.clear_session()
    assert len(sys._session_permissions) == 0


def test_deny_rule_supports_globs():
    sys = PermissionSystem()
    sys.add_rule(PermissionRule(pattern="*delete*", verdict=PermissionMode.DENY))
    result = sys.check(PermissionRequest(action="delete", target="/tmp/file"))
    assert result.verdict == PermissionVerdict.DENIED
