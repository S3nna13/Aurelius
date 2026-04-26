"""Tests for aurelius.inference.compute_scaling_v2."""

from __future__ import annotations

import pytest
import torch
from aurelius.inference.compute_scaling_v2 import (
    BestOfN,
    ComputeScalingOrchestrator,
    InferenceScalingConfig,
    MajorityVoter,
    SelfRefiner,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _scorer(candidates: list[str]) -> torch.Tensor:
    """Return a fixed score for each candidate based on its length."""
    return torch.tensor([float(len(c)) for c in candidates])


def _identity_extractor(s: str) -> str:
    """Return the string unchanged as the 'answer'."""
    return s.strip()


def _make_refiner(n_steps: int = 3) -> SelfRefiner:
    call_counter = {"n": 0}

    def generate_fn(prompt: str) -> str:
        call_counter["n"] += 1
        return f"refined_{call_counter['n']}"

    def critique_fn(prompt: str, response: str) -> str:
        return f"critique of: {response}"

    return SelfRefiner(generate_fn, critique_fn, n_steps=n_steps)


# ---------------------------------------------------------------------------
# InferenceScalingConfig tests
# ---------------------------------------------------------------------------


class TestInferenceScalingConfig:
    def test_defaults(self):
        cfg = InferenceScalingConfig()
        assert cfg.n_samples == 8
        assert cfg.strategy == "best_of_n"
        assert cfg.n_refinement_steps == 3
        assert cfg.temperature == 1.0

    def test_custom_values(self):
        cfg = InferenceScalingConfig(
            n_samples=4, strategy="majority_vote", n_refinement_steps=5, temperature=0.7
        )
        assert cfg.n_samples == 4
        assert cfg.strategy == "majority_vote"
        assert cfg.n_refinement_steps == 5
        assert cfg.temperature == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# BestOfN tests
# ---------------------------------------------------------------------------


class TestBestOfN:
    def test_select_returns_tuple(self):
        bon = BestOfN(_scorer)
        result = bon.select(["hi", "hello", "hey"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_select_returns_correct_candidate_and_idx(self):
        candidates = ["ab", "abcde", "abc"]  # lengths 2, 5, 3 → best is index 1
        bon = BestOfN(_scorer)
        best_str, best_idx = bon.select(candidates)
        assert best_idx == 1
        assert best_str == "abcde"

    def test_select_highest_scored(self):
        candidates = ["x", "longer string", "mid"]
        bon = BestOfN(_scorer)
        best_str, _ = bon.select(candidates)
        assert best_str == "longer string"

    def test_select_batch_returns_one_per_prompt(self):
        bon = BestOfN(_scorer)
        batch = [
            ["a", "bb", "ccc"],  # best: "ccc"
            ["hello", "hi"],  # best: "hello"
        ]
        results = bon.select_batch(batch)
        assert len(results) == 2
        assert results[0] == "ccc"
        assert results[1] == "hello"

    def test_select_batch_empty_inner(self):
        bon = BestOfN(lambda cs: torch.tensor([1.0]))
        results = bon.select_batch([["only one"]])
        assert results == ["only one"]


# ---------------------------------------------------------------------------
# MajorityVoter tests
# ---------------------------------------------------------------------------


class TestMajorityVoter:
    def test_vote_finds_majority(self):
        voter = MajorityVoter(_identity_extractor)
        candidates = ["yes", "no", "yes", "yes", "no"]
        majority, _ = voter.vote(candidates)
        assert majority == "yes"

    def test_vote_returns_counts_dict(self):
        voter = MajorityVoter(_identity_extractor)
        candidates = ["a", "b", "a", "c", "b", "a"]
        _, counts = voter.vote(candidates)
        assert isinstance(counts, dict)
        assert counts["a"] == 3
        assert counts["b"] == 2
        assert counts["c"] == 1

    def test_confidence_correct(self):
        voter = MajorityVoter(_identity_extractor)
        counts: dict[str, int] = {"yes": 6, "no": 4}
        conf = voter.confidence(counts)
        assert conf == pytest.approx(0.6)

    def test_confidence_unanimous(self):
        voter = MajorityVoter(_identity_extractor)
        counts: dict[str, int] = {"yes": 5}
        assert voter.confidence(counts) == pytest.approx(1.0)

    def test_confidence_zero_total(self):
        voter = MajorityVoter(_identity_extractor)
        assert voter.confidence({}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SelfRefiner tests
# ---------------------------------------------------------------------------


class TestSelfRefiner:
    def test_refine_returns_tuple(self):
        refiner = _make_refiner(n_steps=3)
        result = refiner.refine("prompt", "initial")
        assert isinstance(result, tuple) and len(result) == 2

    def test_refine_final_response_is_string(self):
        refiner = _make_refiner(n_steps=2)
        final, _ = refiner.refine("prompt", "initial")
        assert isinstance(final, str)

    def test_refine_history_length(self):
        """History should contain initial + n_steps refined responses."""
        n = 3
        refiner = _make_refiner(n_steps=n)
        _, history = refiner.refine("prompt", "initial")
        assert len(history) == n + 1

    def test_refine_history_starts_with_initial(self):
        refiner = _make_refiner(n_steps=2)
        _, history = refiner.refine("prompt", "initial_response")
        assert history[0] == "initial_response"

    def test_refine_final_equals_last_history(self):
        refiner = _make_refiner(n_steps=3)
        final, history = refiner.refine("prompt", "start")
        assert final == history[-1]

    def test_refine_batch_returns_correct_count(self):
        refiner = _make_refiner(n_steps=1)
        prompts = ["p1", "p2", "p3"]
        initials = ["r1", "r2", "r3"]
        results = refiner.refine_batch(prompts, initials)
        assert len(results) == 3

    def test_refine_batch_all_strings(self):
        refiner = _make_refiner(n_steps=1)
        results = refiner.refine_batch(["a", "b"], ["x", "y"])
        assert all(isinstance(r, str) for r in results)


# ---------------------------------------------------------------------------
# ComputeScalingOrchestrator tests
# ---------------------------------------------------------------------------


class TestComputeScalingOrchestrator:
    def _make_orchestrator(self, strategy: str) -> ComputeScalingOrchestrator:
        cfg = InferenceScalingConfig(strategy=strategy)
        bon = BestOfN(_scorer)
        voter = MajorityVoter(_identity_extractor)
        refiner = _make_refiner(n_steps=cfg.n_refinement_steps)
        return ComputeScalingOrchestrator(cfg, best_of_n=bon, voter=voter, refiner=refiner)

    def test_routes_best_of_n(self):
        orch = self._make_orchestrator("best_of_n")
        candidates = ["short", "longer candidate", "mid"]
        result = orch.run("prompt", candidates)
        assert result == "longer candidate"

    def test_routes_majority_vote(self):
        orch = self._make_orchestrator("majority_vote")
        candidates = ["yes", "no", "yes", "yes"]
        result = orch.run("prompt", candidates)
        assert result == "yes"

    def test_routes_self_refine(self):
        orch = self._make_orchestrator("self_refine")
        result = orch.run("prompt", ["initial"])
        assert isinstance(result, str)
        # self_refine should return something refined, not the raw initial
        assert result != "initial"

    def test_unknown_strategy_raises(self):
        cfg = InferenceScalingConfig(strategy="unknown_strategy")
        orch = ComputeScalingOrchestrator(cfg)
        with pytest.raises(ValueError, match="Unknown strategy"):
            orch.run("prompt", ["candidate"])

    def test_best_of_n_missing_component_raises(self):
        cfg = InferenceScalingConfig(strategy="best_of_n")
        orch = ComputeScalingOrchestrator(cfg)  # no best_of_n provided
        with pytest.raises(ValueError):
            orch.run("prompt", ["a", "b"])

    def test_majority_vote_missing_component_raises(self):
        cfg = InferenceScalingConfig(strategy="majority_vote")
        orch = ComputeScalingOrchestrator(cfg)  # no voter provided
        with pytest.raises(ValueError):
            orch.run("prompt", ["a", "b"])

    def test_self_refine_missing_component_raises(self):
        cfg = InferenceScalingConfig(strategy="self_refine")
        orch = ComputeScalingOrchestrator(cfg)  # no refiner provided
        with pytest.raises(ValueError):
            orch.run("prompt", ["a"])
