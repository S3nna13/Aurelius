"""Tests for src/federation/staleness_handler.py"""

import time

import pytest
import torch

from src.federation.staleness_handler import (
    ClientUpdate,
    StalenessConfig,
    StalenessHandler,
    StalenessPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh() -> StalenessHandler:
    return StalenessHandler()


def make_update(client_id: str = "c0", round_number: int = 0) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        round_number=round_number,
        gradient=torch.ones(4),
        received_at=time.monotonic(),
    )


def cfg(
    policy: StalenessPolicy, max_staleness: int = 5, decay_factor: float = 0.9
) -> StalenessConfig:
    return StalenessConfig(policy=policy, max_staleness=max_staleness, decay_factor=decay_factor)


# ---------------------------------------------------------------------------
# StalenessPolicy enum
# ---------------------------------------------------------------------------


def test_policy_values():
    assert StalenessPolicy.DISCARD == "DISCARD"
    assert StalenessPolicy.DECAY == "DECAY"
    assert StalenessPolicy.BOUNDED_DELAY == "BOUNDED_DELAY"
    assert StalenessPolicy.ALWAYS_ACCEPT == "ALWAYS_ACCEPT"


# ---------------------------------------------------------------------------
# advance_round
# ---------------------------------------------------------------------------


def test_advance_round():
    sh = fresh()
    assert sh.current_round == 0
    sh.advance_round()
    assert sh.current_round == 1
    sh.advance_round()
    assert sh.current_round == 2


# ---------------------------------------------------------------------------
# DISCARD policy
# ---------------------------------------------------------------------------


def test_discard_accept_fresh():
    sh = fresh()
    sh.current_round = 3
    upd = make_update(round_number=3)
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.DISCARD, max_staleness=2))
    assert accepted is True
    assert weight == 1.0


def test_discard_accept_within_max():
    sh = fresh()
    sh.current_round = 5
    upd = make_update(round_number=3)  # staleness = 2
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.DISCARD, max_staleness=2))
    assert accepted is True


def test_discard_reject_beyond_max():
    sh = fresh()
    sh.current_round = 10
    upd = make_update(round_number=3)  # staleness = 7 > max=5
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.DISCARD, max_staleness=5))
    assert accepted is False


# ---------------------------------------------------------------------------
# DECAY policy
# ---------------------------------------------------------------------------


def test_decay_always_accepts():
    sh = fresh()
    sh.current_round = 100
    upd = make_update(round_number=0)  # very stale
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.DECAY, decay_factor=0.9))
    assert accepted is True


def test_decay_weight_decreases_with_staleness():
    sh = fresh()
    sh.current_round = 3

    upd_fresh = make_update(round_number=3)
    upd_stale = make_update(round_number=0)
    _, w_fresh = sh.evaluate(upd_fresh, cfg(StalenessPolicy.DECAY, decay_factor=0.9))
    _, w_stale = sh.evaluate(upd_stale, cfg(StalenessPolicy.DECAY, decay_factor=0.9))
    assert w_fresh > w_stale


def test_decay_weight_formula():
    sh = fresh()
    sh.current_round = 4
    upd = make_update(round_number=2)  # staleness = 2
    _, weight = sh.evaluate(upd, cfg(StalenessPolicy.DECAY, decay_factor=0.5))
    assert weight == pytest.approx(0.25)  # 0.5^2


# ---------------------------------------------------------------------------
# BOUNDED_DELAY policy
# ---------------------------------------------------------------------------


def test_bounded_delay_accept_within():
    sh = fresh()
    sh.current_round = 3
    upd = make_update(round_number=1)  # staleness = 2
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.BOUNDED_DELAY, max_staleness=5))
    assert accepted is True
    assert weight == pytest.approx(1.0 / 3)


def test_bounded_delay_reject_beyond():
    sh = fresh()
    sh.current_round = 10
    upd = make_update(round_number=4)  # staleness = 6 > max=5
    accepted, _ = sh.evaluate(upd, cfg(StalenessPolicy.BOUNDED_DELAY, max_staleness=5))
    assert accepted is False


def test_bounded_delay_zero_staleness_weight_one():
    sh = fresh()
    sh.current_round = 5
    upd = make_update(round_number=5)  # staleness = 0
    _, weight = sh.evaluate(upd, cfg(StalenessPolicy.BOUNDED_DELAY))
    assert weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ALWAYS_ACCEPT policy
# ---------------------------------------------------------------------------


def test_always_accept():
    sh = fresh()
    sh.current_round = 1000
    upd = make_update(round_number=0)
    accepted, weight = sh.evaluate(upd, cfg(StalenessPolicy.ALWAYS_ACCEPT))
    assert accepted is True
    assert weight == 1.0


# ---------------------------------------------------------------------------
# filter_updates
# ---------------------------------------------------------------------------


def test_filter_updates_removes_stale():
    sh = fresh()
    sh.current_round = 10
    updates = [
        make_update("c0", round_number=9),  # staleness 1 – accept
        make_update("c1", round_number=4),  # staleness 6 > 5 – reject
        make_update("c2", round_number=8),  # staleness 2 – accept
    ]
    results = sh.filter_updates(updates, cfg(StalenessPolicy.DISCARD, max_staleness=5))
    assert len(results) == 2
    accepted_ids = {upd.client_id for upd, _ in results}
    assert "c1" not in accepted_ids


def test_filter_updates_returns_weights():
    sh = fresh()
    sh.current_round = 4
    updates = [make_update("c0", round_number=2)]
    results = sh.filter_updates(updates, cfg(StalenessPolicy.DECAY, decay_factor=0.9))
    assert len(results) == 1
    _, weight = results[0]
    assert weight == pytest.approx(0.9**2)
