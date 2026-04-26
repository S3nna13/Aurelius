"""Unit tests for the tiered compaction trigger manager."""

from __future__ import annotations

import pytest

from src.longcontext.compaction_trigger import (
    DEFAULT_TIERS,
    CompactionEvent,
    CompactionTier,
    CompactionTriggerManager,
)


def _identity_compactor(history, ratio):
    # Trivially return a shortened copy to mimic compaction.
    keep = max(1, int(round(len(history) * ratio))) if history else 0
    return history[-keep:] if keep else []


def _build(total_budget: int = 3_000_000) -> CompactionTriggerManager:
    return CompactionTriggerManager(
        tiers=DEFAULT_TIERS,
        compaction_fn=_identity_compactor,
        total_budget=total_budget,
    )


# ---------- defaults -------------------------------------------------------


def test_default_tiers_values():
    assert len(DEFAULT_TIERS) == 2
    fast, slow = DEFAULT_TIERS
    assert (fast.name, fast.threshold_tokens, fast.ratio) == ("fast", 50_000, 0.3)
    assert (slow.name, slow.threshold_tokens, slow.ratio) == ("slow", 200_000, 0.5)


# ---------- observe below threshold ---------------------------------------


def test_observe_below_lowest_threshold_returns_none():
    m = _build()
    assert m.observe(1_000, [{"role": "user", "content": "hi"}]) is None
    assert m.observe(49_999, [{"role": "user", "content": "hi"}]) is None


# ---------- fast tier firing ----------------------------------------------


def test_fast_tier_fires_at_first_threshold():
    m = _build()
    ev = m.observe(50_000, [{"role": "user", "content": "x"}])
    assert isinstance(ev, CompactionEvent)
    assert ev.tier_name == "fast"
    assert ev.before_tokens == 50_000
    assert ev.after_tokens == int(round(50_000 * 0.3))


def test_fast_tier_retrigger_cadence():
    m = _build()
    first = m.observe(50_000, [{"role": "u", "content": "a"}])
    assert first is not None and first.tier_name == "fast"
    # Immediately below next fire mark (50k + 50k = 100k): no fire.
    assert m.observe(99_999, [{"role": "u", "content": "a"}]) is None
    second = m.observe(100_000, [{"role": "u", "content": "a"}])
    assert second is not None and second.tier_name == "fast"


# ---------- slow tier preference ------------------------------------------


def test_slow_tier_preferred_at_slow_threshold():
    m = _build()
    ev = m.observe(200_000, [{"role": "u", "content": "x"}])
    assert ev is not None
    assert ev.tier_name == "slow"
    # slow fired; metrics should reflect exactly one slow and zero fast.
    md = m.metrics()
    assert md["fire_counts"]["slow"] == 1
    assert md["fire_counts"]["fast"] == 0


def test_slow_tier_retrigger_cadence():
    m = _build()
    a = m.observe(200_000, [{"role": "u", "content": "x"}])
    assert a is not None and a.tier_name == "slow"
    # slow next fire at 200k+200k=400k; at 399_999 slow shouldn't fire but fast may.
    b = m.observe(399_999, [{"role": "u", "content": "x"}])
    assert b is not None and b.tier_name == "fast"
    c = m.observe(400_000, [{"role": "u", "content": "x"}])
    assert c is not None and c.tier_name == "slow"


# ---------- total budget ---------------------------------------------------


def test_total_budget_exceeded_raises():
    m = _build(total_budget=100_000)
    with pytest.raises(RuntimeError, match="total_budget"):
        m.observe(100_001, [{"role": "u", "content": "x"}])


# ---------- validation -----------------------------------------------------


def test_ratio_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        CompactionTier("bad", 50_000, -0.1)
    with pytest.raises(ValueError):
        CompactionTier("bad", 50_000, 1.5)
    CompactionTier("edge_lo", 50_000, 0.0)
    CompactionTier("edge_hi", 50_000, 1.0)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        CompactionTier("bad", 0, 0.5)
    with pytest.raises(ValueError):
        CompactionTier("bad", -1, 0.5)


def test_total_budget_must_be_positive():
    with pytest.raises(ValueError):
        CompactionTriggerManager(
            tiers=DEFAULT_TIERS, compaction_fn=_identity_compactor, total_budget=0
        )


def test_tiers_must_be_nonempty_tuple():
    with pytest.raises(ValueError):
        CompactionTriggerManager(tiers=(), compaction_fn=_identity_compactor)


def test_compaction_fn_must_be_callable():
    with pytest.raises(TypeError):
        CompactionTriggerManager(tiers=DEFAULT_TIERS, compaction_fn=42)  # type: ignore[arg-type]


# ---------- reset / metrics ------------------------------------------------


def test_reset_clears_state():
    m = _build()
    m.observe(50_000, [{"role": "u", "content": "x"}])
    m.observe(100_000, [{"role": "u", "content": "x"}])
    assert m.metrics()["fire_counts"]["fast"] == 2
    m.reset()
    md = m.metrics()
    assert md["fire_counts"] == {"fast": 0, "slow": 0}
    assert md["last_fired"] == {"fast": 0, "slow": 0}


def test_metrics_reports_fire_counts_and_shape():
    m = _build()
    m.observe(50_000, [{"role": "u", "content": "x"}])
    m.observe(200_000, [{"role": "u", "content": "x"}])
    md = m.metrics()
    assert md["total_budget"] == 3_000_000
    assert md["total_fires"] == 2
    assert md["fire_counts"] == {"fast": 1, "slow": 1}
    assert [t["name"] for t in md["tiers"]] == ["fast", "slow"]


# ---------- error bubbling -------------------------------------------------


def test_compaction_fn_exception_bubbles_with_context():
    def boom(_history, _ratio):
        raise ValueError("kaboom")

    m = CompactionTriggerManager(tiers=DEFAULT_TIERS, compaction_fn=boom)
    with pytest.raises(RuntimeError, match=r"tier='fast'.*kaboom"):
        m.observe(50_000, [{"role": "u", "content": "x"}])


def test_compaction_fn_bad_return_type_raises():
    def bad(_history, _ratio):
        return "not a list"

    m = CompactionTriggerManager(tiers=DEFAULT_TIERS, compaction_fn=bad)
    with pytest.raises(TypeError):
        m.observe(50_000, [{"role": "u", "content": "x"}])


# ---------- edge cases -----------------------------------------------------


def test_empty_message_history_is_handled():
    m = _build()
    ev = m.observe(50_000, [])
    assert ev is not None
    assert ev.tier_name == "fast"


def test_unicode_messages_are_handled():
    m = _build()
    ev = m.observe(
        50_000,
        [{"role": "user", "content": "安全第一 — déjà vu 🚀"}],
    )
    assert ev is not None
    assert ev.tier_name == "fast"


def test_observe_is_deterministic():
    def det(history, ratio):
        return list(history[: max(1, int(len(history) * ratio))])

    sequence = [50_000, 100_000, 200_000, 400_000]
    history = [{"role": "u", "content": f"msg-{i}"} for i in range(10)]

    def run():
        mgr = CompactionTriggerManager(tiers=DEFAULT_TIERS, compaction_fn=det)
        return [
            mgr.observe(n, history) is not None and mgr.observe.__self__ is mgr  # type: ignore[attr-defined]
            for n in sequence
        ] and [
            (e.tier_name, e.before_tokens, e.after_tokens)
            for e in mgr._events  # noqa: SLF001 -- test inspects record
        ]

    a = run()
    b = run()
    assert a == b
    assert [x[0] for x in a] == ["fast", "fast", "slow", "slow"]


def test_tier_names_must_be_unique():
    with pytest.raises(ValueError):
        CompactionTriggerManager(
            tiers=(
                CompactionTier("dup", 10, 0.5),
                CompactionTier("dup", 20, 0.5),
            ),
            compaction_fn=_identity_compactor,
        )


def test_current_tokens_must_be_nonneg_int():
    m = _build()
    with pytest.raises(ValueError):
        m.observe(-1, [])
    with pytest.raises(ValueError):
        m.observe(-50_000, [])


def test_message_history_must_be_list():
    m = _build()
    with pytest.raises(TypeError):
        m.observe(50_000, "not a list")  # type: ignore[arg-type]
