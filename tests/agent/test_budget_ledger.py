"""Unit tests for src/agent/budget_ledger.py.

Thirty tests covering validation, transactional semantics, determinism,
and registry round-trips.  Pure stdlib.
"""

from __future__ import annotations

import math

import pytest

from src.agent.budget_ledger import (
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
    """Monotonic clock that returns scripted values in nanoseconds."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = int(start_ns)

    def __call__(self) -> int:
        return self._now_ns

    def advance_ms(self, ms: float) -> None:
        self._now_ns += int(ms * 1_000_000)

    def set_ms(self, ms: float) -> None:
        self._now_ns = int(ms * 1_000_000)


@pytest.fixture(autouse=True)
def _clear_ledger_registry():
    """Ensure BUDGET_LEDGER_REGISTRY is empty for each test."""
    BUDGET_LEDGER_REGISTRY.clear()
    yield
    BUDGET_LEDGER_REGISTRY.clear()


# ---------------------------------------------------------------------------
# 1. ResourceLimit validation — empty name
# ---------------------------------------------------------------------------


def test_resource_limit_rejects_empty_name():
    with pytest.raises(LedgerError):
        ResourceLimit(name="", hard_limit=10.0)
    with pytest.raises(LedgerError):
        ResourceLimit(name="   ", hard_limit=10.0)


# ---------------------------------------------------------------------------
# 2. ResourceLimit validation — non-positive hard_limit
# ---------------------------------------------------------------------------


def test_resource_limit_rejects_non_positive_hard_limit():
    with pytest.raises(LedgerError):
        ResourceLimit(name="tokens", hard_limit=0.0)
    with pytest.raises(LedgerError):
        ResourceLimit(name="tokens", hard_limit=-1.0)


# ---------------------------------------------------------------------------
# 3. ResourceLimit validation — soft_fraction boundaries
# ---------------------------------------------------------------------------


def test_resource_limit_rejects_bad_soft_fraction():
    with pytest.raises(LedgerError):
        ResourceLimit(name="tokens", hard_limit=10.0, soft_fraction=0.0)
    with pytest.raises(LedgerError):
        ResourceLimit(name="tokens", hard_limit=10.0, soft_fraction=-0.1)
    with pytest.raises(LedgerError):
        ResourceLimit(name="tokens", hard_limit=10.0, soft_fraction=1.01)


def test_resource_limit_allows_soft_fraction_one():
    lim = ResourceLimit(name="tokens", hard_limit=10.0, soft_fraction=1.0)
    assert lim.soft_fraction == 1.0


# ---------------------------------------------------------------------------
# 4. BudgetLedger construction — empty limits
# ---------------------------------------------------------------------------


def test_ledger_requires_non_empty_limits():
    with pytest.raises(LedgerError):
        BudgetLedger(limits=())


def test_ledger_requires_tuple_limits():
    with pytest.raises(LedgerError):
        BudgetLedger(limits=[ResourceLimit("tokens", 10.0)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. BudgetLedger construction — duplicate names
# ---------------------------------------------------------------------------


def test_ledger_rejects_duplicate_resource_names():
    with pytest.raises(LedgerError):
        BudgetLedger(
            limits=(
                ResourceLimit("tokens", 10.0),
                ResourceLimit("tokens", 20.0),
            )
        )


# ---------------------------------------------------------------------------
# 6. charge — negative amount
# ---------------------------------------------------------------------------


def test_charge_rejects_negative_amount():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    with pytest.raises(LedgerError):
        ledger.charge("tokens", -1.0)


def test_charge_rejects_nan_amount():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    with pytest.raises(LedgerError):
        ledger.charge("tokens", float("nan"))


# ---------------------------------------------------------------------------
# 7. charge — unknown resource
# ---------------------------------------------------------------------------


def test_charge_rejects_unknown_resource():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    with pytest.raises(LedgerError):
        ledger.charge("ghost", 1.0)


# ---------------------------------------------------------------------------
# 8. charge — under soft threshold returns OK
# ---------------------------------------------------------------------------


def test_charge_below_soft_returns_ok():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    assert ledger.charge("tokens", 50.0) == BudgetSeverity.OK


# ---------------------------------------------------------------------------
# 9. charge — strictly above soft threshold returns SOFT
# ---------------------------------------------------------------------------


def test_charge_above_soft_returns_soft():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    sev = ledger.charge("tokens", 85.0)
    assert sev == BudgetSeverity.SOFT


# ---------------------------------------------------------------------------
# 10. charge — exactly at hard limit returns SOFT (boundary)
# ---------------------------------------------------------------------------


def test_charge_exactly_at_hard_is_soft():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    sev = ledger.charge("tokens", 100.0)
    assert sev == BudgetSeverity.SOFT
    assert ledger.spent("tokens") == 100.0


# ---------------------------------------------------------------------------
# 11. charge — strictly over hard raises and state is unchanged
# ---------------------------------------------------------------------------


def test_charge_over_hard_raises_and_leaves_state_unchanged():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    assert ledger.charge("tokens", 50.0) == BudgetSeverity.OK
    with pytest.raises(BudgetExhaustedError):
        ledger.charge("tokens", 60.0)  # 50 + 60 = 110 > 100
    assert ledger.spent("tokens") == 50.0
    assert ledger.remaining("tokens") == 50.0


# ---------------------------------------------------------------------------
# 12. soft-fraction boundary — at exactly soft_fraction * hard, severity is OK
# ---------------------------------------------------------------------------


def test_charge_at_soft_boundary_is_ok_strict_gt():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    # Exactly at 80 (the soft threshold) — strict-greater-than semantics
    # mean severity is still OK.
    sev = ledger.charge("tokens", 80.0)
    assert sev == BudgetSeverity.OK
    assert ledger.severity("tokens") == BudgetSeverity.OK


# ---------------------------------------------------------------------------
# 13. remaining decreases monotonically
# ---------------------------------------------------------------------------


def test_remaining_decreases_monotonically():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    before = ledger.remaining("tokens")
    ledger.charge("tokens", 10.0)
    after1 = ledger.remaining("tokens")
    ledger.charge("tokens", 25.0)
    after2 = ledger.remaining("tokens")
    assert before == 100.0
    assert after1 == 90.0
    assert after2 == 65.0
    assert before >= after1 >= after2


# ---------------------------------------------------------------------------
# 14. snapshot structure and values
# ---------------------------------------------------------------------------


def test_snapshot_structure():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.5),),
        clock_ns=clock,
    )
    ledger.charge("tokens", 60.0)
    clock.advance_ms(5.0)
    snap = ledger.snapshot()
    assert isinstance(snap, LedgerSnapshot)
    assert snap.resources["tokens"] == 60.0
    assert snap.limits["tokens"] == 100.0
    assert snap.severities["tokens"] == BudgetSeverity.SOFT
    assert snap.remaining["tokens"] == 40.0
    assert snap.step_count == 0
    assert snap.started_ns == 0
    assert snap.elapsed_ms == pytest.approx(5.0)
    # wall_ms is auto-inserted
    assert "wall_ms" in snap.resources
    assert snap.limits["wall_ms"] == math.inf


# ---------------------------------------------------------------------------
# 15. worst_severity across resources
# ---------------------------------------------------------------------------


def test_worst_severity_aggregates():
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 100.0, soft_fraction=0.5),
            ResourceLimit("cost_usd", 10.0, soft_fraction=0.5),
        )
    )
    assert ledger.worst_severity() == BudgetSeverity.OK
    ledger.charge("tokens", 60.0)  # SOFT on tokens
    assert ledger.worst_severity() == BudgetSeverity.SOFT
    # cost_usd still OK, tokens SOFT → worst is SOFT
    ledger.charge("cost_usd", 1.0)
    assert ledger.worst_severity() == BudgetSeverity.SOFT


# ---------------------------------------------------------------------------
# 16. is_exhausted — False by default on fresh ledger
# ---------------------------------------------------------------------------


def test_fresh_ledger_not_exhausted():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    assert ledger.is_exhausted() is False
    assert ledger.worst_severity() == BudgetSeverity.OK


# ---------------------------------------------------------------------------
# 17. is_exhausted — True after wall_ms silent exhaustion via fake clock
# ---------------------------------------------------------------------------


def test_wall_ms_silent_hard_exhaustion_via_step():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 100.0),
            ResourceLimit("wall_ms", 1_000.0, soft_fraction=0.8),
        ),
        clock_ns=clock,
    )
    assert ledger.is_exhausted() is False

    # Fast-forward past hard limit (1 s).
    clock.set_ms(2_000.0)
    # step() must NOT raise even though wall_ms is now strictly over hard.
    ledger.step()
    assert ledger.severity("wall_ms") == BudgetSeverity.HARD
    assert ledger.is_exhausted() is True
    assert ledger.worst_severity() == BudgetSeverity.HARD


# ---------------------------------------------------------------------------
# 18. wall_ms auto-inserted when not supplied
# ---------------------------------------------------------------------------


def test_wall_ms_auto_inserted_with_inf_limit():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    assert "wall_ms" in ledger.resource_names()
    snap = ledger.snapshot()
    assert snap.limits["wall_ms"] == math.inf
    # Auto-inserted wall_ms never trips.
    assert ledger.severity("wall_ms") == BudgetSeverity.OK


# ---------------------------------------------------------------------------
# 19. context manager updates wall_ms on exit
# ---------------------------------------------------------------------------


def test_context_manager_updates_wall_ms_on_exit():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 100.0),
            ResourceLimit("wall_ms", 10_000.0),
        ),
        clock_ns=clock,
    )
    with ledger as active:
        assert active is ledger
        clock.advance_ms(123.0)
    # On __exit__ wall_ms was refreshed from the fake clock.
    assert ledger.spent("wall_ms") == pytest.approx(123.0)


# ---------------------------------------------------------------------------
# 20. reset restores zero state
# ---------------------------------------------------------------------------


def test_reset_restores_zero_state():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0),),
        clock_ns=clock,
    )
    ledger.charge("tokens", 50.0)
    ledger.step()
    clock.advance_ms(500.0)
    assert ledger.spent("tokens") == 50.0
    assert ledger.step_count == 1

    clock.set_ms(1_000.0)
    ledger.reset()
    assert ledger.spent("tokens") == 0.0
    assert ledger.step_count == 0
    assert ledger.started_ns == 1_000 * 1_000_000
    assert ledger.worst_severity() == BudgetSeverity.OK


# ---------------------------------------------------------------------------
# 21. determinism with fake clock
# ---------------------------------------------------------------------------


def test_deterministic_wall_ms_with_fake_clock():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(
            ResourceLimit("tokens", 100.0),
            ResourceLimit("wall_ms", 10_000.0),
        ),
        clock_ns=clock,
    )
    clock.advance_ms(250.0)
    ledger.step()
    assert ledger.spent("wall_ms") == pytest.approx(250.0)

    clock.advance_ms(250.0)
    ledger.step()
    assert ledger.spent("wall_ms") == pytest.approx(500.0)
    assert ledger.step_count == 2


# ---------------------------------------------------------------------------
# 22. register_ledger / get_ledger / list_ledgers round-trip
# ---------------------------------------------------------------------------


def test_registry_round_trip():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    assert list_ledgers() == ()
    register_ledger("session-a", ledger)
    assert list_ledgers() == ("session-a",)
    assert get_ledger("session-a") is ledger


# ---------------------------------------------------------------------------
# 23. duplicate registration rejected
# ---------------------------------------------------------------------------


def test_registry_rejects_duplicate_name():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    register_ledger("session", ledger)
    with pytest.raises(LedgerError):
        register_ledger("session", ledger)


# ---------------------------------------------------------------------------
# 24. bad registry name raises
# ---------------------------------------------------------------------------


def test_registry_rejects_bad_names():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    with pytest.raises(LedgerError):
        register_ledger("", ledger)
    with pytest.raises(LedgerError):
        register_ledger("has space", ledger)
    with pytest.raises(LedgerError):
        register_ledger("bad/slash", ledger)


def test_registry_rejects_non_ledger():
    with pytest.raises(LedgerError):
        register_ledger("bad", "not-a-ledger")  # type: ignore[arg-type]


def test_get_ledger_unknown_raises():
    with pytest.raises(LedgerError):
        get_ledger("does-not-exist")


# ---------------------------------------------------------------------------
# 25. make_default_ledger — five expected resources
# ---------------------------------------------------------------------------


def test_make_default_ledger_has_expected_resources():
    ledger = make_default_ledger()
    names = set(ledger.resource_names())
    assert {"tokens", "wall_ms", "cost_usd", "tool_calls", "errors"} <= names
    assert ledger.remaining("tokens") == 128_000.0
    assert ledger.remaining("cost_usd") == 1.0
    assert ledger.remaining("tool_calls") == 32.0
    assert ledger.remaining("errors") == 8.0
    assert ledger.remaining("wall_ms") == 120_000.0


# ---------------------------------------------------------------------------
# 26. overflow protection — very large charge raises BudgetExhaustedError
# ---------------------------------------------------------------------------


def test_overflow_charge_raises_exhausted_error():
    ledger = make_default_ledger()
    with pytest.raises(BudgetExhaustedError):
        ledger.charge("tokens", 1e30)
    # And state is still clean: no tokens were charged.
    assert ledger.spent("tokens") == 0.0


# ---------------------------------------------------------------------------
# 27. snapshot().elapsed_ms is non-negative and grows with fake-clock advance
# ---------------------------------------------------------------------------


def test_snapshot_elapsed_ms_grows():
    clock = _FakeClock(start_ns=0)
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0),),
        clock_ns=clock,
    )
    snap0 = ledger.snapshot()
    assert snap0.elapsed_ms >= 0.0
    clock.advance_ms(10.0)
    snap1 = ledger.snapshot()
    assert snap1.elapsed_ms >= snap0.elapsed_ms
    assert snap1.elapsed_ms == pytest.approx(10.0)
    clock.advance_ms(15.0)
    snap2 = ledger.snapshot()
    assert snap2.elapsed_ms == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 28. remaining on unknown resource raises LedgerError
# ---------------------------------------------------------------------------


def test_remaining_unknown_resource_raises():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 100.0),))
    with pytest.raises(LedgerError):
        ledger.remaining("ghost")
    with pytest.raises(LedgerError):
        ledger.severity("ghost")


# ---------------------------------------------------------------------------
# 29. charge accumulates across multiple calls and severity progresses
# ---------------------------------------------------------------------------


def test_charge_progression_ok_soft_then_raise():
    ledger = BudgetLedger(
        limits=(ResourceLimit("tokens", 100.0, soft_fraction=0.8),)
    )
    assert ledger.charge("tokens", 40.0) == BudgetSeverity.OK
    assert ledger.charge("tokens", 30.0) == BudgetSeverity.OK  # 70, still OK
    assert ledger.charge("tokens", 20.0) == BudgetSeverity.SOFT  # 90, over 80
    with pytest.raises(BudgetExhaustedError):
        ledger.charge("tokens", 20.0)  # 110 > 100
    # Pre-raise state preserved.
    assert ledger.spent("tokens") == 90.0


# ---------------------------------------------------------------------------
# 30. clock_ns validation
# ---------------------------------------------------------------------------


def test_bad_clock_rejected():
    with pytest.raises(LedgerError):
        BudgetLedger(
            limits=(ResourceLimit("tokens", 100.0),),
            clock_ns="not-callable",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 31. BudgetExhaustedError carries informative message fields
# ---------------------------------------------------------------------------


def test_budget_exhausted_message_fields():
    ledger = BudgetLedger(limits=(ResourceLimit("tokens", 10.0),))
    ledger.charge("tokens", 5.0)
    with pytest.raises(BudgetExhaustedError) as exc_info:
        ledger.charge("tokens", 10.0)
    msg = str(exc_info.value)
    assert "tokens" in msg
    assert "10" in msg  # requested
    assert "5" in msg  # remaining


# ---------------------------------------------------------------------------
# 32. LedgerSnapshot fields round-trip through all configured resources
# ---------------------------------------------------------------------------


def test_snapshot_covers_all_resources():
    ledger = make_default_ledger()
    snap = ledger.snapshot()
    for name in ledger.resource_names():
        assert name in snap.resources
        assert name in snap.limits
        assert name in snap.severities
        assert name in snap.remaining
