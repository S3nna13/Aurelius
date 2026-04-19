"""Integration test: rejection-sampling DPO pair synthesis end-to-end."""

from __future__ import annotations

from itertools import count

from src.data.rejection_sampling_data import PreferencePair, RejectionSampler


def test_exposed_via_src_data_submodule():
    # The module is reachable via the src.data package path.
    import src.data.rejection_sampling_data as mod

    assert mod.RejectionSampler is RejectionSampler
    assert mod.PreferencePair is PreferencePair


def test_end_to_end_five_prompts():
    # Deterministic fake policy: each call returns prompt + :: + monotonic
    # counter. Reward = trailing int, so highest index wins.
    c = count()

    def generate_fn(prompt: str) -> str:
        return f"{prompt}::{next(c)}"

    def reward_fn(prompt: str, response: str) -> float:
        return float(response.rsplit("::", 1)[-1])

    sampler = RejectionSampler(
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n_samples=4,
        min_margin=0.5,
    )

    prompts = [
        "write a quicksort in python",
        "explain the halting problem",
        "design a REST API for todos",
        "prove sqrt(2) irrational",
        "implement binary search",
    ]
    pairs = sampler.sample_pairs(prompts)

    # One pair per prompt (all margins are well above 0.5 since rewards are
    # distinct integers spanning at least n_samples-1 apart).
    assert len(pairs) == 5
    assert [p.prompt for p in pairs] == prompts

    for p in pairs:
        assert isinstance(p, PreferencePair)
        assert p.chosen_reward > p.rejected_reward
        assert p.margin == p.chosen_reward - p.rejected_reward
        assert p.margin >= 0.5
        assert p.chosen != p.rejected
        assert p.chosen.startswith(p.prompt + "::")
        assert p.rejected.startswith(p.prompt + "::")

    # The full candidate pool produced exactly 5 prompts * 4 samples = 20
    # generator calls.
    assert next(c) == 20


def test_end_to_end_all_pairs_for_one_prompt():
    c = count()

    def generate_fn(prompt: str) -> str:
        return f"{prompt}::{next(c)}"

    def reward_fn(prompt: str, response: str) -> float:
        return float(response.rsplit("::", 1)[-1])

    sampler = RejectionSampler(
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n_samples=4,
    )
    pairs = sampler.sample_all_pairs("refactor this function")
    # C(4, 2) = 6
    assert len(pairs) == 6
    for p in pairs:
        assert p.chosen_reward > p.rejected_reward
        assert p.prompt == "refactor this function"
