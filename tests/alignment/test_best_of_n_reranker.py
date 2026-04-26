"""Unit tests for Best-of-N reranker."""

from __future__ import annotations

import math
import random

import pytest

from src.alignment.best_of_n_reranker import BestOfNReranker, BoNCandidate

# ---------------------------------------------------------------- fixtures


def make_generator(responses):
    """Return a stateful generate_fn that yields ``responses`` in order."""
    it = iter(responses)

    def gen(prompt: str) -> str:
        return next(it)

    return gen


def make_reward(mapping):
    """Reward_fn that returns ``mapping[response]``."""

    def reward(prompt: str, response: str) -> float:
        return mapping[response]

    return reward


# --------------------------------------------------------------- tests (13)


def test_rerank_returns_n_candidates_sorted_desc():
    gen = make_generator(["a", "b", "c", "d"])
    rew = make_reward({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.3})
    rr = BestOfNReranker(gen, rew, n=4)
    ranked = rr.rerank("p")
    assert len(ranked) == 4
    rewards = [c.reward for c in ranked]
    assert rewards == sorted(rewards, reverse=True)
    assert ranked[0].response == "b"


def test_best_returns_top1():
    gen = make_generator(["x", "y", "z"])
    rew = make_reward({"x": 0.2, "y": 0.7, "z": 0.5})
    rr = BestOfNReranker(gen, rew, n=3)
    top = rr.best("p")
    assert isinstance(top, BoNCandidate)
    assert top.response == "y"
    assert top.rank == 0


def test_weighted_vote_groups_by_answer():
    # Three candidates answer "42", two answer "7". "42" group has higher sum.
    responses = ["ans 42", "ans 7", "ans 42", "ans 42", "ans 7"]
    rewards = {"ans 42": 0.3, "ans 7": 0.9}
    # Each response's reward is looked up once per response; but we need
    # per-call rewards, not per-string. Use a fresh reward fn that returns
    # a fixed reward per string (so 3 * 0.3 = 0.9 vs 2 * 0.9 = 1.8 -> "7").
    gen = make_generator(responses)
    rew = make_reward(rewards)
    rr = BestOfNReranker(gen, rew, n=5, aggregation="weighted_vote")

    def extract(r: str) -> str:
        return r.split()[-1]

    winner = rr.weighted_vote("p", extract)
    assert winner == "7"


def test_ties_broken_by_generation_index():
    gen = make_generator(["first", "second", "third"])
    # All identical reward -> tie-break by index -> "first" wins.
    rew = make_reward({"first": 0.5, "second": 0.5, "third": 0.5})
    rr = BestOfNReranker(gen, rew, n=3)
    ranked = rr.rerank("p")
    assert [c.response for c in ranked] == ["first", "second", "third"]
    assert ranked[0].rank == 0


def test_n_equals_one():
    gen = make_generator(["only"])
    rew = make_reward({"only": 0.42})
    rr = BestOfNReranker(gen, rew, n=1)
    ranked = rr.rerank("p")
    assert len(ranked) == 1
    assert ranked[0].response == "only"
    assert ranked[0].rank == 0


def test_n_zero_raises():
    with pytest.raises(ValueError, match="n must be >= 1"):
        BestOfNReranker(lambda p: "x", lambda p, r: 0.0, n=0)


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="aggregation"):
        BestOfNReranker(lambda p: "x", lambda p, r: 0.0, n=2, aggregation="bogus")


def test_generate_fn_exception_skips_candidate(caplog):
    calls = {"i": 0}

    def gen(prompt: str) -> str:
        calls["i"] += 1
        if calls["i"] == 2:
            raise RuntimeError("boom")
        return f"resp_{calls['i']}"

    def rew(prompt: str, response: str) -> float:
        return float(response.split("_")[-1])

    rr = BestOfNReranker(gen, rew, n=4)
    with caplog.at_level("WARNING"):
        ranked = rr.rerank("p")
    # 4 calls, 1 failed -> 3 candidates.
    assert len(ranked) == 3
    assert all(c.response.startswith("resp_") for c in ranked)
    assert any("generate_fn raised" in rec.message for rec in caplog.records)


def test_reward_fn_exception_candidate_gets_neg_inf():
    gen = make_generator(["a", "b", "c"])

    def rew(prompt: str, response: str) -> float:
        if response == "b":
            raise RuntimeError("reward boom")
        return {"a": 0.1, "c": 0.9}[response]

    rr = BestOfNReranker(gen, rew, n=3)
    ranked = rr.rerank("p")
    assert len(ranked) == 3
    last = ranked[-1]
    assert last.response == "b"
    assert last.reward == float("-inf")
    assert last.rank == 2


def test_determinism_with_fixed_seed():
    def build():
        rng = random.Random(123)
        pool = ["alpha", "beta", "gamma", "delta", "epsilon"]

        def gen(prompt: str) -> str:
            return rng.choice(pool)

        def rew(prompt: str, response: str) -> float:
            return float(len(response))

        return BestOfNReranker(gen, rew, n=6)

    r1 = [c.response for c in build().rerank("p")]
    r2 = [c.response for c in build().rerank("p")]
    assert r1 == r2


def test_empty_prompt_works():
    gen = make_generator(["r1", "r2"])
    rew = make_reward({"r1": 1.0, "r2": 2.0})
    rr = BestOfNReranker(gen, rew, n=2)
    top = rr.best("")
    assert top.response == "r2"


def test_weighted_vote_all_different_returns_highest_reward():
    # Each answer unique -> sum == single reward -> best-of-N wins.
    responses = ["ans A", "ans B", "ans C", "ans D"]
    rewards = {"ans A": 0.1, "ans B": 0.9, "ans C": 0.5, "ans D": 0.7}
    gen = make_generator(responses)
    rew = make_reward(rewards)
    rr = BestOfNReranker(gen, rew, n=4, aggregation="weighted_vote")
    winner = rr.weighted_vote("p", lambda r: r.split()[-1])
    assert winner == "B"


def test_rerank_returns_BoNCandidate_with_populated_ranks():
    gen = make_generator(["x", "y", "z"])
    rew = make_reward({"x": 0.1, "y": 0.5, "z": 0.3})
    rr = BestOfNReranker(gen, rew, n=3)
    ranked = rr.rerank("p")
    assert all(isinstance(c, BoNCandidate) for c in ranked)
    assert [c.rank for c in ranked] == [0, 1, 2]


def test_rank_monotonically_increasing():
    gen = make_generator([f"r{i}" for i in range(6)])
    rew = make_reward({f"r{i}": float(i) * 0.11 for i in range(6)})
    rr = BestOfNReranker(gen, rew, n=6)
    ranked = rr.rerank("p")
    ranks = [c.rank for c in ranked]
    assert ranks == list(range(6))
    # And strictly increasing.
    assert all(b > a for a, b in zip(ranks, ranks[1:]))


def test_nan_reward_treated_as_failure():
    gen = make_generator(["ok", "bad"])

    def rew(prompt: str, response: str) -> float:
        return math.nan if response == "bad" else 0.5

    rr = BestOfNReranker(gen, rew, n=2)
    ranked = rr.rerank("p")
    assert ranked[0].response == "ok"
    assert ranked[-1].response == "bad"
    assert ranked[-1].reward == float("-inf")
