"""Integration tests for src/agent/budget_ledger.

Covers public re-exports, default-factory wiring, registry round-trip, a
full-loop spending scenario, and deterministic severity escalation using
an injectable clock.
"""

from __future__ import annotations

import pytest

from src.agent import (
    BUDGET_LEDGER_REGISTRY,
    BudgetExhaustedError,
    BudgetLedger,
    BudgetSeverity,
    LedgerError,
    LedgerSnapshot,
    ResourceLimit,
    get_ledger,
    list_ledgers,
    make_default_ledger,
    register_ledger,
)


class _FakeClock:
    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = int(start_ns)

    def __call__(self) -> int:
        return self._now_ns

    def advance_ms(self, ms: float) -> None:
        self._now_ns += int(ms * 1_000_000)

    def set_ms(self, ms: float) -> None:
        self._now_ns = int(ms * 1_000_000)


@pytest.fixture(autouse=True)
def _clear_registry():
    BUDGET_LEDGER_REGISTRY.clear()
    yield
    BUDGET_LEDGER_REGISTRY.clear()


# ---------------------------------------------------------------------------
# 1. Public re-exports are reachable from src.agent
# ---------------------------------------------------------------------------


def test_public_reexports_available():
    assert BudgetLedger is not None
    assert ResourceLimit is not None
    assert BudgetSeverity is not None
    assert BudgetExhaustedError is not None
    assert LedgerError is not None
    assert LedgerSnapshot is not None
    assert callable(make_default_ledger)
    assert callable(register_ledger)
    assert callable(get_ledger)
    assert callable(list_ledgers)
    assert isinstance(BUDGET_LEDGER_REGISTRY, dict)


# ---------------------------------------------------------------------------
# 2. make_default_ledger() has the five canonical resources
# ---------------------------------------------------------------------------


def test_default_ledger_has_five_resources():
    ledger = make_default_ledger()
    names = set(ledger.resource_names())
    for expected in ("tokens", "wall_ms", "cost_usd", "tool_calls", "errors"):
        assert expected in names, f"expected {expected!r} in {names}"


# ---------------------------------------------------------------------------
# 3. Register + retrieve round-trip with identity assertion
# ---------------------------------------------------------------------------


def test_registry_identity_round_trip():
    ledger = make_default_ledger()
    register_ledger("worker-7", ledger)
    assert "worker-7" in list_ledgers()
    assert get_ledger("worker-7") is ledger


# ---------------------------------------------------------------------------
# 4. Combined scenario — charge tokens across a loop and inspect snapshot
# ---------------------------------------------------------------------------


def test_combined_loop_scenario():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 1_000.0, soft_fraction=0.8),
            ResourceLimit("tool_calls", 5.0, soft_fraction=0.8),
            ResourceLimit("errors", 3.0, soft_fraction=0.5),
        ),
        clock_ns=clock,
    )

    severities = []
    for turn in range(3):
        sev = ledger.charge("tokens", 200.0)
        severities.append(sev)
        ledger.charge("tool_calls", 1.0)
        clock.advance_ms(100.0)
        ledger.step()

    assert ledger.spent("tokens") == pytest.approx(600.0)
    assert ledger.spent("tool_calls") == 3.0
    assert ledger.spent("wall_ms") == pytest.approx(300.0)
    assert ledger.step_count == 3
    # All turns stayed in OK territory (max 600/1000 below 0.8*1000).
    assert all(s == BudgetSeverity.OK for s in severities)

    # Pushing one more into SOFT territory.
    sev = ledger.charge("tokens", 250.0)  # 850 > 800 → SOFT
    assert sev == BudgetSeverity.SOFT

    snap = ledger.snapshot()
    assert snap.resources["tokens"] == pytest.approx(850.0)
    assert snap.remaining["tokens"] == pytest.approx(150.0)
    assert snap.severities["tokens"] == BudgetSeverity.SOFT
    assert snap.step_count == 3

    # Attempting to overspend raises without mutating state.
    with pytest.raises(BudgetExhaustedError):
        ledger.charge("tokens", 500.0)
    assert ledger.spent("tokens") == pytest.approx(850.0)


# ---------------------------------------------------------------------------
# 5. worst_severity transitions OK → SOFT → HARD deterministically
# ---------------------------------------------------------------------------


def test_worst_severity_transitions_ok_soft_hard():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 100.0, soft_fraction=0.5),
            ResourceLimit("wall_ms", 1_000.0, soft_fraction=0.8),
        ),
        clock_ns=clock,
    )
    assert ledger.worst_severity() == BudgetSeverity.OK

    # OK → SOFT by spending past tokens' soft threshold.
    ledger.charge("tokens", 60.0)  # 60 > 50 → SOFT
    assert ledger.worst_severity() == BudgetSeverity.SOFT
    assert ledger.is_exhausted() is False

    # SOFT → HARD by fast-forwarding the fake clock past wall_ms hard limit.
    clock.set_ms(2_000.0)
    ledger.step()  # silent wall_ms update
    assert ledger.severity("wall_ms") == BudgetSeverity.HARD
    assert ledger.worst_severity() == BudgetSeverity.HARD
    assert ledger.is_exhausted() is True


# ---------------------------------------------------------------------------
# 6. make_default_ledger — charge + snapshot end-to-end
# ---------------------------------------------------------------------------


def test_default_ledger_end_to_end():
    clock = _FakeClock(start_ns=0)
    ledger = make_default_ledger(
        token_budget=1_000,
        wall_ms_budget=10_000,
        cost_usd_budget=5.0,
        tool_call_budget=4,
        error_budget=2,
        clock_ns=clock,
    )
    ledger.charge("tokens", 100)
    ledger.charge("cost_usd", 0.5)
    ledger.charge("tool_calls", 1)
    clock.advance_ms(42.0)
    ledger.step()

    snap = ledger.snapshot()
    assert snap.resources["tokens"] == 100.0
    assert snap.resources["cost_usd"] == 0.5
    assert snap.resources["tool_calls"] == 1.0
    assert snap.resources["wall_ms"] == pytest.approx(42.0)
    assert snap.resources["errors"] == 0.0
    assert snap.step_count == 1
    # Nothing is SOFT or HARD yet in this scenario.
    for name in ("tokens", "cost_usd", "tool_calls", "wall_ms", "errors"):
        assert snap.severities[name] == BudgetSeverity.OK
