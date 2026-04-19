"""Unit tests for src.data.rejection_sampling_data."""

from __future__ import annotations

from itertools import count
from math import comb

import pytest

from src.data.rejection_sampling_data import PreferencePair, RejectionSampler


# ---------------------------------------------------------------------------
# deterministic fakes
# ---------------------------------------------------------------------------
def make_counter_generate():
    """generate_fn that returns f'{prompt}::{i}' with i monotonically increasing."""
    c = count()
    def gen(prompt: str) -> str:
        return f"{prompt}::{next(c)}"
    return gen


def length_reward(prompt: str, response: str) -> float:
    """Reward = length of response. Deterministic."""
    return float(len(response))


def index_reward(prompt: str, response: str) -> float:
    """Reward = trailing integer in response 'prompt::N'."""
    return float(response.rsplit("::", 1)[-1])


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_sample_pairs_one_per_prompt():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=4,
    )
    prompts = ["a", "b", "c"]
    pairs = sampler.sample_pairs(prompts)
    assert len(pairs) == 3
    assert [p.prompt for p in pairs] == prompts
    assert all(isinstance(p, PreferencePair) for p in pairs)


def test_sample_top_bottom_none_if_margin_below_min():
    # all candidates identical -> margin 0
    sampler = RejectionSampler(
        generate_fn=lambda p: "same",
        reward_fn=lambda p, r: 1.0,
        n_samples=4,
        min_margin=0.5,
    )
    assert sampler.sample_top_bottom("x") is None


def test_sample_all_pairs_returns_c_n_2():
    n = 5
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=n,
    )
    pairs = sampler.sample_all_pairs("p")
    # All candidates distinct rewards -> no ties dropped
    assert len(pairs) == comb(n, 2)


def test_margin_is_chosen_minus_rejected():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=4,
    )
    pair = sampler.sample_top_bottom("p")
    assert pair is not None
    assert pair.margin == pytest.approx(pair.chosen_reward - pair.rejected_reward)


def test_chosen_reward_ge_rejected():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=6,
    )
    pair = sampler.sample_top_bottom("p")
    assert pair is not None
    assert pair.chosen_reward >= pair.rejected_reward
    for pp in sampler.sample_all_pairs("q"):
        assert pp.chosen_reward >= pp.rejected_reward


def test_empty_prompts_returns_empty():
    sampler = RejectionSampler(
        generate_fn=lambda p: "x",
        reward_fn=lambda p, r: 1.0,
    )
    assert sampler.sample_pairs([]) == []


def test_n_samples_1_returns_none():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=1,
    )
    assert sampler.sample_top_bottom("p") is None
    assert sampler.sample_all_pairs("p") == []
    assert sampler.sample_pairs(["a", "b"]) == []


def test_invalid_n_samples_raises():
    with pytest.raises(ValueError):
        RejectionSampler(
            generate_fn=lambda p: "x",
            reward_fn=lambda p, r: 0.0,
            n_samples=0,
        )
    with pytest.raises(ValueError):
        RejectionSampler(
            generate_fn=lambda p: "x",
            reward_fn=lambda p, r: 0.0,
            n_samples=-3,
        )


def test_invalid_min_margin_raises():
    with pytest.raises(ValueError):
        RejectionSampler(
            generate_fn=lambda p: "x",
            reward_fn=lambda p, r: 0.0,
            min_margin=-0.01,
        )


def test_reward_fn_exception_skips_candidate():
    calls = {"n": 0}

    def flaky_reward(prompt, response):
        calls["n"] += 1
        # Fail on 2nd candidate (index 1)
        idx = int(response.rsplit("::", 1)[-1])
        if idx % 10 == 1:
            raise RuntimeError("boom")
        return float(idx)

    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=flaky_reward,
        n_samples=4,
    )
    pair = sampler.sample_top_bottom("p")
    assert pair is not None
    # The candidate with idx==1 should have been skipped; so rewards exclude 1.
    assert pair.rejected_reward != 1.0
    assert pair.chosen_reward != 1.0


def test_determinism():
    # Two independent samplers with identical deterministic funcs should
    # produce identical output.
    def make():
        return RejectionSampler(
            generate_fn=make_counter_generate(),
            reward_fn=length_reward,
            n_samples=5,
        )

    s1 = make()
    s2 = make()
    prompts = ["alpha", "beta", "gamma"]
    out1 = s1.sample_pairs(prompts)
    out2 = s2.sample_pairs(prompts)
    assert out1 == out2


def test_determinism_with_ties():
    # All candidates have equal reward -> sample_top_bottom should give None
    # (margin 0 < default min_margin 0.0 is allowed, but best==worst index
    # tie-break must be deterministic; with equal rewards, best and worst
    # resolve to index 0 and thus None).
    sampler = RejectionSampler(
        generate_fn=lambda p: "same",
        reward_fn=lambda p, r: 7.0,
        n_samples=4,
    )
    # With identical rewards, best-tiebreak picks smallest index (0) and
    # worst-tiebreak also picks smallest index (0) -> None.
    assert sampler.sample_top_bottom("p") is None


def test_max_candidates_caps_generation():
    call_count = {"n": 0}

    def gen(prompt):
        call_count["n"] += 1
        return f"{prompt}::{call_count['n']}"

    sampler = RejectionSampler(
        generate_fn=gen,
        reward_fn=length_reward,
        n_samples=10,
        max_candidates=3,
    )
    sampler.sample_top_bottom("p")
    assert call_count["n"] == 3


def test_large_prompts_batch():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=3,
    )
    prompts = [f"p{i}" for i in range(100)]
    pairs = sampler.sample_pairs(prompts)
    assert len(pairs) == 100
    for p in pairs:
        assert p.chosen_reward > p.rejected_reward


def test_preference_pair_fields_populated():
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=4,
    )
    pair = sampler.sample_top_bottom("hello")
    assert pair is not None
    assert pair.prompt == "hello"
    assert isinstance(pair.chosen, str) and pair.chosen.startswith("hello::")
    assert isinstance(pair.rejected, str) and pair.rejected.startswith("hello::")
    assert isinstance(pair.chosen_reward, float)
    assert isinstance(pair.rejected_reward, float)
    assert isinstance(pair.margin, float)
    assert pair.chosen != pair.rejected


def test_min_margin_filters_in_sample_pairs():
    # Rewards are indices 0..3; margin between top/bottom per prompt is 3.
    sampler = RejectionSampler(
        generate_fn=make_counter_generate(),
        reward_fn=index_reward,
        n_samples=4,
        min_margin=100.0,  # impossible
    )
    out = sampler.sample_pairs(["a", "b"])
    assert out == []


def test_invalid_max_candidates_raises():
    with pytest.raises(ValueError):
        RejectionSampler(
            generate_fn=lambda p: "x",
            reward_fn=lambda p, r: 0.0,
            max_candidates=0,
        )
