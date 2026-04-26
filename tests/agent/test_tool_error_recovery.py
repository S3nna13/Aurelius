"""Unit tests for :mod:`src.agent.tool_error_recovery`."""

from __future__ import annotations

import pytest

from src.agent.tool_error_recovery import (
    ErrorClassifier,
    RecoveringDispatcher,
    RecoveryDecision,
    RecoveryPolicy,
    RecoveryStrategy,
    decide,
)
from src.agent.tool_registry_dispatcher import ToolInvocationResult


def _fail(name: str, error: str) -> ToolInvocationResult:
    return ToolInvocationResult(
        name=name, ok=False, value=None, error=error, duration_ms=1.0, truncated=False
    )


def _ok(name: str, value) -> ToolInvocationResult:
    return ToolInvocationResult(
        name=name, ok=True, value=value, error=None, duration_ms=1.0, truncated=False
    )


# -- ErrorClassifier --------------------------------------------------------


def test_error_classifier_buckets():
    c = ErrorClassifier.classify
    assert c("timeout:exceeded_5.0s") == "timeout"
    assert c("rate_limited") == "rate_limit"
    assert c("unknown_tool") == "unknown_tool"
    assert c("schema_error:missing_required:foo") == "validation"
    assert c("invalid_arguments:not_a_dict") == "validation"
    assert c("tool_argument_error:missing 1 required") == "validation"
    assert c("permission denied: admin only") == "permission_denied"
    assert c("forbidden") == "permission_denied"
    assert c("something weird happened") == "internal_error"
    assert c(None) == "internal_error"


# -- decide -----------------------------------------------------------------


def test_decide_timeout_retries_with_backoff():
    policy = RecoveryPolicy(max_retries=3, backoff_base=0.5, backoff_factor=2.0)
    d = decide(_fail("net", "timeout:exceeded_5s"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.RETRY_BACKOFF
    assert d.delay_seconds == pytest.approx(0.5)
    d2 = decide(_fail("net", "timeout"), policy, attempt_count=2)
    assert d2.delay_seconds == pytest.approx(1.0)


def test_decide_validation_requests_modified_args():
    policy = RecoveryPolicy(max_retries=3)
    d = decide(_fail("calc", "schema_error:missing_required:x"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.RETRY_MODIFIED_ARGS


def test_decide_validation_aborts_when_exhausted():
    policy = RecoveryPolicy(max_retries=2)
    d = decide(_fail("calc", "schema_error:missing_required:x"), policy, attempt_count=2)
    assert d.strategy == RecoveryStrategy.ABORT


def test_decide_unknown_tool_uses_fallback_when_configured():
    policy = RecoveryPolicy(fallback_map={"old_search": "new_search"})
    d = decide(_fail("old_search", "unknown_tool"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.FALLBACK_TOOL
    assert d.fallback_tool == "new_search"


def test_decide_unknown_tool_aborts_without_fallback():
    policy = RecoveryPolicy()
    d = decide(_fail("old_search", "unknown_tool"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.ABORT


def test_decide_permission_denied_escalates():
    policy = RecoveryPolicy(max_retries=5)
    d = decide(_fail("fs", "permission denied"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.ESCALATE


def test_decide_aborts_at_max_retries():
    policy = RecoveryPolicy(max_retries=3)
    d = decide(_fail("net", "timeout"), policy, attempt_count=3)
    assert d.strategy == RecoveryStrategy.ABORT


def test_decide_backoff_capped_at_max_delay():
    policy = RecoveryPolicy(max_retries=10, backoff_base=1.0, backoff_factor=10.0, max_delay=2.5)
    d = decide(_fail("net", "timeout"), policy, attempt_count=5)
    assert d.delay_seconds == pytest.approx(2.5)


def test_decide_abort_on_list_overrides():
    policy = RecoveryPolicy(max_retries=5, abort_on=["budget:"])
    d = decide(_fail("t", "budget:total_calls_exhausted"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.ABORT


def test_decide_escalate_on_list_overrides():
    policy = RecoveryPolicy(max_retries=5, escalate_on=["policy_violation"])
    d = decide(_fail("t", "policy_violation:x"), policy, attempt_count=1)
    assert d.strategy == RecoveryStrategy.ESCALATE


def test_decide_rejects_successful_result():
    with pytest.raises(ValueError):
        decide(_ok("t", 1), RecoveryPolicy(), attempt_count=1)


def test_decide_deterministic():
    policy = RecoveryPolicy(max_retries=3, backoff_base=0.5, backoff_factor=2.0)
    r = _fail("net", "timeout")
    a = decide(r, policy, attempt_count=2)
    b = decide(r, policy, attempt_count=2)
    assert a == b


# -- Policy / decision validation ------------------------------------------


def test_policy_rejects_negative_retries():
    with pytest.raises(ValueError):
        RecoveryPolicy(max_retries=-1)


def test_policy_rejects_bad_backoff_factor():
    with pytest.raises(ValueError):
        RecoveryPolicy(backoff_factor=0.5)


def test_decision_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        RecoveryDecision(strategy="teleport", reason="nope")


# -- RecoveringDispatcher ---------------------------------------------------


class _FakeDispatcher:
    def __init__(self, script):
        # script: list of (result_factory) or callables taking (name,args)
        self.script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def dispatch(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        item = self.script.pop(0)
        if callable(item):
            return item(name, arguments)
        return item


def test_recovering_dispatcher_retries_then_succeeds():
    inner = _FakeDispatcher(
        [
            _fail("t", "timeout"),
            _fail("t", "timeout"),
            _ok("t", 42),
        ]
    )
    sleeps: list[float] = []
    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=5, backoff_base=0.1, backoff_factor=2.0),
        sleep=sleeps.append,
    )
    result = disp.dispatch("t", {"x": 1})
    assert result.ok is True
    assert result.value == 42
    assert len(inner.calls) == 3
    assert sleeps == [pytest.approx(0.1), pytest.approx(0.2)]


def test_recovering_dispatcher_exhausts_retries():
    inner = _FakeDispatcher([_fail("t", "timeout")] * 4)
    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=3, backoff_base=0.0),
        sleep=lambda s: None,
    )
    result = disp.dispatch("t", {"x": 1})
    assert result.ok is False
    assert result.error is not None and "timeout" in result.error
    # Three attempts: 2 retries + final abort attempt. The third call
    # returns failure and decide() at attempt_count=3 -> abort.
    assert len(inner.calls) == 3
    assert any(d.strategy == RecoveryStrategy.ABORT for _, d in disp.recovery_history())


def test_recovering_dispatcher_uses_fallback():
    inner = _FakeDispatcher(
        [
            _fail("old", "unknown_tool"),
            _ok("new", "ok"),
        ]
    )
    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(fallback_map={"old": "new"}),
        sleep=lambda s: None,
    )
    result = disp.dispatch("old", {})
    assert result.ok is True
    assert result.value == "ok"
    assert [c[0] for c in inner.calls] == ["old", "new"]


def test_recovering_dispatcher_retry_modified_args_uses_mutator():
    inner = _FakeDispatcher(
        [
            _fail("calc", "schema_error:missing_required:y"),
            _ok("calc", 7),
        ]
    )

    def mutator(name, args, result):
        patched = dict(args)
        patched["y"] = 3
        return patched

    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=3),
        args_mutator=mutator,
        sleep=lambda s: None,
    )
    result = disp.dispatch("calc", {"x": 1})
    assert result.ok is True
    assert inner.calls[0][1] == {"x": 1}
    assert inner.calls[1][1] == {"x": 1, "y": 3}


def test_recovering_dispatcher_retry_modified_args_without_mutator_aborts():
    inner = _FakeDispatcher([_fail("calc", "schema_error:missing_required:y")])
    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=3),
        sleep=lambda s: None,
    )
    result = disp.dispatch("calc", {"x": 1})
    assert result.ok is False
    assert len(inner.calls) == 1


def test_recovering_dispatcher_escalate_returns_failure():
    inner = _FakeDispatcher([_fail("fs", "permission denied")])
    disp = RecoveringDispatcher(inner, RecoveryPolicy(), sleep=lambda s: None)
    result = disp.dispatch("fs", {})
    assert result.ok is False
    hist = disp.recovery_history()
    assert hist and hist[-1][1].strategy == RecoveryStrategy.ESCALATE


def test_recovering_dispatcher_requires_recovery_policy():
    inner = _FakeDispatcher([])
    with pytest.raises(TypeError):
        RecoveringDispatcher(inner, policy="strict")  # type: ignore[arg-type]
