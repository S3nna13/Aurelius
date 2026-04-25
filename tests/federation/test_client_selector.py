"""Tests for src/federation/client_selector.py."""

from __future__ import annotations

import pytest

from src.federation.client_selector import (
    ClientProfile,
    ClientSelector,
    SelectionStrategy,
    CLIENT_SELECTOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_profile(
    client_id: str,
    compute_score: float = 1.0,
    data_size: int = 0,
    last_selected_round: int = -1,
) -> ClientProfile:
    return ClientProfile(
        client_id=client_id,
        compute_score=compute_score,
        data_size=data_size,
        last_selected_round=last_selected_round,
    )


def register_n(selector: ClientSelector, n: int) -> list[ClientProfile]:
    profiles = [make_profile(f"c{i}", compute_score=float(i + 1)) for i in range(n)]
    for p in profiles:
        selector.register(p)
    return profiles


# ---------------------------------------------------------------------------
# ClientProfile dataclass
# ---------------------------------------------------------------------------

class TestClientProfile:
    def test_required_field_stored(self):
        p = ClientProfile(client_id="c1")
        assert p.client_id == "c1"

    def test_compute_score_default(self):
        p = ClientProfile(client_id="c1")
        assert p.compute_score == 1.0

    def test_data_size_default(self):
        p = ClientProfile(client_id="c1")
        assert p.data_size == 0

    def test_last_selected_round_default(self):
        p = ClientProfile(client_id="c1")
        assert p.last_selected_round == -1

    def test_explicit_values(self):
        p = ClientProfile(client_id="c2", compute_score=3.5, data_size=1000, last_selected_round=5)
        assert p.compute_score == 3.5
        assert p.data_size == 1000
        assert p.last_selected_round == 5


# ---------------------------------------------------------------------------
# register / client_count
# ---------------------------------------------------------------------------

class TestRegisterAndCount:
    def test_initial_count_zero(self):
        sel = ClientSelector()
        assert sel.client_count() == 0

    def test_register_increases_count(self):
        sel = ClientSelector()
        sel.register(make_profile("c1"))
        assert sel.client_count() == 1

    def test_register_multiple(self):
        sel = ClientSelector()
        register_n(sel, 5)
        assert sel.client_count() == 5

    def test_register_overwrites_same_id(self):
        sel = ClientSelector()
        sel.register(make_profile("c1", compute_score=1.0))
        sel.register(make_profile("c1", compute_score=9.9))
        assert sel.client_count() == 1


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------

class TestDeregister:
    def test_deregister_existing_returns_true(self):
        sel = ClientSelector()
        sel.register(make_profile("c1"))
        assert sel.deregister("c1") is True

    def test_deregister_existing_decreases_count(self):
        sel = ClientSelector()
        sel.register(make_profile("c1"))
        sel.deregister("c1")
        assert sel.client_count() == 0

    def test_deregister_missing_returns_false(self):
        sel = ClientSelector()
        assert sel.deregister("ghost") is False

    def test_deregister_then_reregister(self):
        sel = ClientSelector()
        sel.register(make_profile("c1"))
        sel.deregister("c1")
        sel.register(make_profile("c1"))
        assert sel.client_count() == 1


# ---------------------------------------------------------------------------
# select — general
# ---------------------------------------------------------------------------

class TestSelectGeneral:
    def test_select_zero_clients_returns_empty(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 5)
        result = sel.select(0, round_idx=0, seed=42)
        assert result == []

    def test_select_more_than_available_returns_all(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 3)
        result = sel.select(100, round_idx=0, seed=42)
        assert len(result) == 3

    def test_select_updates_last_selected_round(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 4)
        selected = sel.select(2, round_idx=7, seed=0)
        for p in selected:
            assert p.last_selected_round == 7

    def test_select_returns_list(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 3)
        result = sel.select(2, seed=1)
        assert isinstance(result, list)

    def test_select_returns_client_profiles(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 3)
        result = sel.select(2, seed=1)
        assert all(isinstance(p, ClientProfile) for p in result)


# ---------------------------------------------------------------------------
# select — RANDOM
# ---------------------------------------------------------------------------

class TestSelectRandom:
    def test_random_count_correct(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 10)
        result = sel.select(4, seed=0)
        assert len(result) == 4

    def test_random_seed_deterministic(self):
        sel1 = ClientSelector(SelectionStrategy.RANDOM)
        sel2 = ClientSelector(SelectionStrategy.RANDOM)
        profiles = [make_profile(f"c{i}") for i in range(10)]
        for p in profiles:
            sel1.register(make_profile(p.client_id))
            sel2.register(make_profile(p.client_id))
        r1 = [p.client_id for p in sel1.select(4, seed=99)]
        r2 = [p.client_id for p in sel2.select(4, seed=99)]
        assert r1 == r2

    def test_random_no_duplicates(self):
        sel = ClientSelector(SelectionStrategy.RANDOM)
        register_n(sel, 10)
        result = sel.select(5, seed=42)
        ids = [p.client_id for p in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# select — POWER_OF_CHOICE
# ---------------------------------------------------------------------------

class TestSelectPowerOfChoice:
    def test_poc_count_correct(self):
        sel = ClientSelector(SelectionStrategy.POWER_OF_CHOICE)
        register_n(sel, 10)
        result = sel.select(3, seed=0)
        assert len(result) == 3

    def test_poc_selects_high_compute_score(self):
        sel = ClientSelector(SelectionStrategy.POWER_OF_CHOICE)
        # Register clients: one very high scorer
        sel.register(make_profile("low1", compute_score=0.1))
        sel.register(make_profile("low2", compute_score=0.1))
        sel.register(make_profile("low3", compute_score=0.1))
        sel.register(make_profile("high", compute_score=100.0))
        # With 2k=8 candidates from 4 clients, all are candidates;
        # top 1 by compute_score must be "high"
        result = sel.select(1, seed=0)
        assert result[0].client_id == "high"

    def test_poc_no_duplicates(self):
        sel = ClientSelector(SelectionStrategy.POWER_OF_CHOICE)
        register_n(sel, 8)
        result = sel.select(4, seed=7)
        ids = [p.client_id for p in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# select — RESOURCE_AWARE
# ---------------------------------------------------------------------------

class TestSelectResourceAware:
    def test_resource_aware_count_correct(self):
        sel = ClientSelector(SelectionStrategy.RESOURCE_AWARE)
        register_n(sel, 6)
        result = sel.select(3)
        assert len(result) == 3

    def test_resource_aware_sorted_descending(self):
        sel = ClientSelector(SelectionStrategy.RESOURCE_AWARE)
        sel.register(make_profile("c1", compute_score=1.0))
        sel.register(make_profile("c2", compute_score=5.0))
        sel.register(make_profile("c3", compute_score=3.0))
        result = sel.select(3)
        scores = [p.compute_score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_resource_aware_top_k_by_score(self):
        sel = ClientSelector(SelectionStrategy.RESOURCE_AWARE)
        sel.register(make_profile("high", compute_score=10.0))
        sel.register(make_profile("mid", compute_score=5.0))
        sel.register(make_profile("low", compute_score=1.0))
        result = sel.select(2)
        ids = {p.client_id for p in result}
        assert "high" in ids
        assert "mid" in ids
        assert "low" not in ids


# ---------------------------------------------------------------------------
# select — ROUND_ROBIN
# ---------------------------------------------------------------------------

class TestSelectRoundRobin:
    def test_round_robin_count_correct(self):
        sel = ClientSelector(SelectionStrategy.ROUND_ROBIN)
        register_n(sel, 6)
        result = sel.select(3, round_idx=0)
        assert len(result) == 3

    def test_round_robin_different_rounds_differ(self):
        sel = ClientSelector(SelectionStrategy.ROUND_ROBIN)
        register_n(sel, 6)
        r0 = {p.client_id for p in sel.select(2, round_idx=0)}
        r1 = {p.client_id for p in sel.select(2, round_idx=1)}
        # With 6 clients, k=2, round 0 picks indices 0,1 and round 1 picks 2,3
        assert r0 != r1

    def test_round_robin_wraps_around(self):
        sel = ClientSelector(SelectionStrategy.ROUND_ROBIN)
        register_n(sel, 3)
        # k=1, 3 clients: round 0 → idx 0, round 3 → idx 0 again
        r0 = sel.select(1, round_idx=0)[0].client_id
        r3 = sel.select(1, round_idx=3)[0].client_id
        assert r0 == r3

    def test_round_robin_no_duplicates(self):
        sel = ClientSelector(SelectionStrategy.ROUND_ROBIN)
        register_n(sel, 5)
        result = sel.select(3, round_idx=0)
        ids = [p.client_id for p in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# CLIENT_SELECTOR_REGISTRY
# ---------------------------------------------------------------------------

class TestClientSelectorRegistry:
    def test_registry_exists(self):
        assert CLIENT_SELECTOR_REGISTRY is not None

    def test_registry_has_default(self):
        assert "default" in CLIENT_SELECTOR_REGISTRY

    def test_registry_default_is_class(self):
        assert CLIENT_SELECTOR_REGISTRY["default"] is ClientSelector

    def test_registry_default_callable(self):
        cls = CLIENT_SELECTOR_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, ClientSelector)
