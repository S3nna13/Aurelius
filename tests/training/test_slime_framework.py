"""Unit tests for the Slime RL Framework — GLM-5 §4 (arXiv:2602.15763)."""

from __future__ import annotations

import pytest

from src.training.slime_framework import (
    SlimeTask,
    SlimeTaskRouter,
    _overlap_score,
    make_default_router,
)


@pytest.fixture
def router() -> SlimeTaskRouter:
    return make_default_router()


@pytest.fixture
def swe(router: SlimeTaskRouter) -> SlimeTask:
    return router.route("swe")


@pytest.fixture
def terminal(router: SlimeTaskRouter) -> SlimeTask:
    return router.route("terminal")


@pytest.fixture
def search(router: SlimeTaskRouter) -> SlimeTask:
    return router.route("search")


# ---------------------------------------------------------------------------
# 1. Routing
# ---------------------------------------------------------------------------


def test_route_swe_returns_task(router: SlimeTaskRouter) -> None:
    task = router.route("swe")
    assert isinstance(task, SlimeTask)
    assert task.name == "swe"


def test_route_terminal_returns_task(router: SlimeTaskRouter) -> None:
    task = router.route("terminal")
    assert isinstance(task, SlimeTask)
    assert task.name == "terminal"


def test_route_search_returns_task(router: SlimeTaskRouter) -> None:
    task = router.route("search")
    assert isinstance(task, SlimeTask)
    assert task.name == "search"


def test_route_unknown_raises(router: SlimeTaskRouter) -> None:
    with pytest.raises(ValueError, match="unknown"):
        router.route("unknown")


# ---------------------------------------------------------------------------
# 2. SWE task — exact match
# ---------------------------------------------------------------------------


def test_swe_verifier_exact_match(swe: SlimeTask) -> None:
    assert swe.verify("abc", "abc") is True


def test_swe_verifier_no_match(swe: SlimeTask) -> None:
    assert swe.verify("abc", "xyz") is False


def test_swe_reward_match(swe: SlimeTask) -> None:
    assert swe.reward("abc", "abc") == 1.0


def test_swe_reward_no_match(swe: SlimeTask) -> None:
    assert swe.reward("abc", "xyz") == 0.0


def test_swe_verifier_strips_whitespace(swe: SlimeTask) -> None:
    assert swe.verify("  abc  ", "abc") is True


# ---------------------------------------------------------------------------
# 3. Terminal task — substring match
# ---------------------------------------------------------------------------


def test_terminal_verifier_substring(terminal: SlimeTask) -> None:
    assert terminal.verify("result: ok", "ok") is True


def test_terminal_verifier_no_substring(terminal: SlimeTask) -> None:
    assert terminal.verify("result: ok", "fail") is False


def test_terminal_reward_match(terminal: SlimeTask) -> None:
    assert terminal.reward("the answer is yes", "yes") == 1.0


def test_terminal_reward_no_match(terminal: SlimeTask) -> None:
    assert terminal.reward("the answer is yes", "no") == 0.0


# ---------------------------------------------------------------------------
# 4. Search task — word-overlap reward
# ---------------------------------------------------------------------------


def test_search_reward_partial(search: SlimeTask) -> None:
    score = search.reward("the quick brown", "the quick fox")
    assert 0.0 < score < 1.0


def test_search_reward_full(search: SlimeTask) -> None:
    assert search.reward("fox", "fox") == 1.0


def test_search_verifier_any_overlap(search: SlimeTask) -> None:
    assert search.verify("hello world", "world peace") is True


def test_search_verifier_no_overlap(search: SlimeTask) -> None:
    assert search.verify("apple banana", "orange grape") is False


# ---------------------------------------------------------------------------
# 5. Router meta-behaviour
# ---------------------------------------------------------------------------


def test_register_custom_task(router: SlimeTaskRouter) -> None:
    custom = SlimeTask(
        name="custom",
        verifier=lambda c, t: len(c) == len(t),
        reward_fn=lambda c, t: 1.0 if len(c) == len(t) else 0.0,
    )
    router.register_task(custom)
    retrieved = router.route("custom")
    assert retrieved.name == "custom"
    assert retrieved.verify("abc", "xyz") is True
    assert retrieved.reward("abc", "xy") == 0.0


def test_registered_types_sorted(router: SlimeTaskRouter) -> None:
    types = router.registered_types()
    assert types == sorted(types)
    assert set(types) == {"swe", "terminal", "search"}


def test_len_default() -> None:
    assert len(make_default_router()) == 3


# ---------------------------------------------------------------------------
# 6. Overlap score edge cases
# ---------------------------------------------------------------------------


def test_overlap_score_empty_target() -> None:
    assert _overlap_score("hello", "") == 0.0


def test_overlap_score_both_empty() -> None:
    assert _overlap_score("", "") == 1.0
