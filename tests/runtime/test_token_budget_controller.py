from __future__ import annotations

import pytest

from src.runtime.token_budget_controller import BudgetConfig, TokenBudgetController


@pytest.fixture()
def ctrl():
    return TokenBudgetController()


@pytest.fixture()
def tight():
    cfg = BudgetConfig(
        max_input_tokens=100,
        max_output_tokens=50,
        max_total_tokens=120,
    )
    return TokenBudgetController(config=cfg)


def test_create_session_returns_zero_state(ctrl):
    state = ctrl.create_session("s1")
    assert state.session_id == "s1"
    assert state.input_tokens_used == 0
    assert state.output_tokens_used == 0
    assert state.total_tokens_used == 0
    assert state.cost_usd == 0.0
    assert state.hard_limit_hit is False


def test_list_sessions_includes_created(ctrl):
    ctrl.create_session("a")
    ctrl.create_session("b")
    assert set(ctrl.list_sessions()) == {"a", "b"}


def test_record_usage_increments_counters(ctrl):
    ctrl.create_session("s")
    state = ctrl.record_usage("s", input_tokens=100, output_tokens=50)
    assert state.input_tokens_used == 100
    assert state.output_tokens_used == 50
    assert state.total_tokens_used == 150


def test_record_usage_accumulates_across_calls(ctrl):
    ctrl.create_session("s")
    ctrl.record_usage("s", 200, 100)
    state = ctrl.record_usage("s", 300, 200)
    assert state.input_tokens_used == 500
    assert state.output_tokens_used == 300
    assert state.total_tokens_used == 800


def test_record_usage_computes_cost(ctrl):
    ctrl.create_session("s")
    state = ctrl.record_usage("s", 1_000_000, 1_000_000)
    expected = 1_000_000 * 0.000003 + 1_000_000 * 0.000015
    assert abs(state.cost_usd - expected) < 1e-9


def test_record_usage_sets_hard_limit_on_input_exceeded(tight):
    tight.create_session("s")
    state = tight.record_usage("s", 101, 0)
    assert state.hard_limit_hit is True


def test_record_usage_sets_hard_limit_on_output_exceeded(tight):
    tight.create_session("s")
    state = tight.record_usage("s", 0, 51)
    assert state.hard_limit_hit is True


def test_record_usage_sets_hard_limit_on_total_exceeded(tight):
    tight.create_session("s")
    state = tight.record_usage("s", 70, 70)
    assert state.hard_limit_hit is True


def test_check_budget_allows_within_limits(tight):
    tight.create_session("s")
    allowed, reason = tight.check_budget("s", requested_input=50, requested_output=20)
    assert allowed is True
    assert reason == "ok"


def test_check_budget_denies_input_overflow(tight):
    tight.create_session("s")
    allowed, reason = tight.check_budget("s", requested_input=101)
    assert allowed is False
    assert "input" in reason


def test_check_budget_denies_output_overflow(tight):
    tight.create_session("s")
    allowed, reason = tight.check_budget("s", requested_output=51)
    assert allowed is False
    assert "output" in reason


def test_check_budget_denies_total_overflow(tight):
    tight.create_session("s")
    allowed, reason = tight.check_budget("s", requested_input=70, requested_output=51)
    assert allowed is False


def test_get_remaining_initial(tight):
    tight.create_session("s")
    rem = tight.get_remaining("s")
    assert rem["input_remaining"] == 100
    assert rem["output_remaining"] == 50
    assert rem["total_remaining"] == 120
    assert rem["cost_usd"] == 0.0
    assert rem["budget_pct_used"] == 0.0


def test_get_remaining_after_usage(tight):
    tight.create_session("s")
    tight.record_usage("s", 60, 30)
    rem = tight.get_remaining("s")
    assert rem["input_remaining"] == 40
    assert rem["output_remaining"] == 20
    assert rem["total_remaining"] == 30


def test_get_remaining_does_not_go_negative(tight):
    tight.create_session("s")
    tight.record_usage("s", 200, 200)
    rem = tight.get_remaining("s")
    assert rem["input_remaining"] == 0
    assert rem["output_remaining"] == 0
    assert rem["total_remaining"] == 0


def test_reset_session_zeroes_counters(ctrl):
    ctrl.create_session("s")
    ctrl.record_usage("s", 500, 200)
    state = ctrl.reset_session("s")
    assert state.input_tokens_used == 0
    assert state.cost_usd == 0.0
    assert state.hard_limit_hit is False


def test_total_cost_sums_sessions(ctrl):
    ctrl.create_session("a")
    ctrl.create_session("b")
    ctrl.record_usage("a", 1_000_000, 0)
    ctrl.record_usage("b", 0, 1_000_000)
    expected = 1_000_000 * 0.000003 + 1_000_000 * 0.000015
    assert abs(ctrl.total_cost() - expected) < 1e-9


def test_prune_sessions_removes_excess(ctrl):
    for i in range(10):
        ctrl.create_session(str(i))
    removed = ctrl.prune_sessions(max_sessions=5)
    assert removed == 5
    assert len(ctrl.list_sessions()) == 5


def test_prune_sessions_noop_when_under_limit(ctrl):
    ctrl.create_session("x")
    removed = ctrl.prune_sessions(max_sessions=100)
    assert removed == 0


def test_unknown_session_raises(ctrl):
    with pytest.raises(KeyError):
        ctrl.record_usage("ghost", 10, 10)


def test_budget_pct_used_is_correct(tight):
    tight.create_session("s")
    tight.record_usage("s", 60, 0)
    rem = tight.get_remaining("s")
    assert abs(rem["budget_pct_used"] - 50.0) < 1e-6
