"""Integration tests for multi-sample voting via ``src.inference``."""

from __future__ import annotations

import re

import src.inference as inference_pkg


def test_exposed_via_inference_package():
    assert hasattr(inference_pkg, "MultiSampleVoter")
    assert hasattr(inference_pkg, "VoteResult")
    assert "MultiSampleVoter" in inference_pkg.__all__
    assert "VoteResult" in inference_pkg.__all__


def test_prior_inference_entries_intact():
    # Prior public entries must still be exported.
    for name in (
        "BatchStep",
        "ContinuousBatchingScheduler",
        "InferenceRequest",
        "JSONDecoderState",
        "JSONMaskBuilder",
        "SCHEDULER_REGISTRY",
        "is_valid_json_prefix",
    ):
        assert name in inference_pkg.__all__, f"{name} missing from __all__"
        assert hasattr(inference_pkg, name), f"{name} missing from module"


def test_end_to_end_majority_vote_on_five_math_samples():
    def last_int(t: str) -> str:
        m = re.findall(r"-?\d+", t)
        return m[-1] if m else ""

    voter = inference_pkg.MultiSampleVoter(
        answer_extractor=last_int,
        strategy="majority",
    )
    samples = [
        "Let's compute 6*7 = 42.",
        "Step-by-step: 6+6+6+6+6+6+6 = 42",
        "I think the answer is 41.",
        "Plug into the formula, get 42 as final.",
        "final boxed answer 42",
    ]
    result = voter.vote(samples)
    assert isinstance(result, inference_pkg.VoteResult)
    assert result.selected == "42"
    assert result.votes["42"] == 4.0
    assert result.votes["41"] == 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == 0.8


def test_end_to_end_usc_on_five_samples():
    voter = inference_pkg.MultiSampleVoter(strategy="usc")
    samples = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over the lazy dog",
        "the quick brown fox hops over the lazy dog",
        "the quick brown fox jumps over a sleepy dog",
        "completely unrelated string of characters xyz",
    ]
    result = voter.vote(samples)
    assert result.strategy == "usc"
    assert result.selected != samples[-1]
    assert result.selected in samples[:4]
