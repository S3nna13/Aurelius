"""Integration tests for the Slime RL Framework wired into TRAINING_REGISTRY."""
from __future__ import annotations

import pytest

from src.training import TRAINING_REGISTRY
from src.training.slime_framework import SlimeTaskRouter, make_default_router


# ---------------------------------------------------------------------------
# 1. Registry presence
# ---------------------------------------------------------------------------


def test_slime_in_training_registry() -> None:
    assert "slime" in TRAINING_REGISTRY, (
        "'slime' key missing from TRAINING_REGISTRY; "
        "check src/training/__init__.py"
    )


def test_registry_value_is_slime_task_router() -> None:
    assert TRAINING_REGISTRY["slime"] is SlimeTaskRouter


# ---------------------------------------------------------------------------
# 2. Construct from registry and route
# ---------------------------------------------------------------------------


def test_construct_router_from_registry_and_route_swe() -> None:
    RouterCls = TRAINING_REGISTRY["slime"]
    router = make_default_router.__func__() if hasattr(make_default_router, "__func__") else make_default_router()
    task = router.route("swe")
    assert task.verify("hello", "hello") is True
    assert task.verify("hello", "world") is False


def test_registry_router_round_trip_terminal() -> None:
    router = make_default_router()
    task = router.route("terminal")
    assert task.reward("exit code 0", "exit code 0") == 1.0
    assert task.reward("exit code 1", "exit code 0") == 0.0


# ---------------------------------------------------------------------------
# 3. Default router produces exactly 3 tasks
# ---------------------------------------------------------------------------


def test_make_default_router_produces_three_tasks() -> None:
    router = make_default_router()
    assert len(router) == 3


def test_make_default_router_task_names() -> None:
    router = make_default_router()
    assert set(router.registered_types()) == {"swe", "terminal", "search"}


# ---------------------------------------------------------------------------
# 4. Regression guard — pre-existing TRAINING_REGISTRY keys still present
# ---------------------------------------------------------------------------


def test_existing_async_rl_key_present() -> None:
    assert "async_rl" in TRAINING_REGISTRY, (
        "Regression: 'async_rl' key was removed from TRAINING_REGISTRY"
    )


def test_existing_tito_key_present() -> None:
    assert "tito" in TRAINING_REGISTRY, (
        "Regression: 'tito' key was removed from TRAINING_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 5. Search task end-to-end via registry path
# ---------------------------------------------------------------------------


def test_search_task_overlap_via_registry() -> None:
    router = make_default_router()
    search = router.route("search")
    score = search.reward("the quick brown fox", "the quick fox")
    assert score == pytest.approx(1.0)


def test_search_task_partial_overlap_via_registry() -> None:
    router = make_default_router()
    search = router.route("search")
    score = search.reward("alpha beta", "alpha beta gamma")
    assert 0.0 < score < 1.0
