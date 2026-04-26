"""Tests for inference-time compute scaling (compute_scaling.py)."""

from __future__ import annotations

import random

import pytest

from src.inference.compute_scaling import (
    ComputeScaler,
    PassAtK,
    ScalingConfig,
    best_of_n,
    compute_response_diversity,
    majority_vote,
    normalize_scores,
    weighted_majority_vote,
)

# ---------------------------------------------------------------------------
# Deterministic mock generate function
# ---------------------------------------------------------------------------


def make_mock_generate(seed: int = 42):
    """Return a generate_fn that picks randomly from ["A", "B", "C"]."""
    rng = random.Random(seed)

    def generate_fn(prompt: str) -> str:  # noqa: ARG001
        return rng.choice(["A", "B", "C"])

    return generate_fn


# ---------------------------------------------------------------------------
# ScalingConfig defaults
# ---------------------------------------------------------------------------


def test_scaling_config_defaults():
    cfg = ScalingConfig()
    assert cfg.n_samples == 8
    assert cfg.temperature == 0.8
    assert cfg.aggregation == "majority_vote"


def test_scaling_config_custom():
    cfg = ScalingConfig(n_samples=4, temperature=1.0, aggregation="best_of_n")
    assert cfg.n_samples == 4
    assert cfg.temperature == 1.0
    assert cfg.aggregation == "best_of_n"


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


def test_majority_vote_correct():
    assert majority_vote(["A", "B", "A", "A", "B"]) == "A"


def test_majority_vote_tie_break_first():
    # "B" appears first but same count as "A"; "B" should win tie-break
    result = majority_vote(["B", "A"])
    assert result == "B"


def test_majority_vote_all_same():
    assert majority_vote(["X", "X", "X"]) == "X"


def test_majority_vote_single():
    assert majority_vote(["Z"]) == "Z"


# ---------------------------------------------------------------------------
# weighted_majority_vote
# ---------------------------------------------------------------------------


def test_weighted_majority_vote_equal_weights():
    # With equal weights, highest-count response should win (same as majority_vote)
    responses = ["A", "A", "B", "B", "B"]
    weights = [1.0] * 5
    assert weighted_majority_vote(responses, weights) == "B"


def test_weighted_majority_vote_unequal_weights():
    # A gets weight 10.0, B gets 1.0 + 1.0 = 2.0; A should win despite fewer occurrences
    responses = ["A", "B", "B"]
    weights = [10.0, 1.0, 1.0]
    assert weighted_majority_vote(responses, weights) == "A"


# ---------------------------------------------------------------------------
# best_of_n
# ---------------------------------------------------------------------------


def test_best_of_n_correct():
    responses = ["low", "medium", "high"]
    scores = [0.1, 0.5, 0.9]
    assert best_of_n(responses, scores) == "high"


def test_best_of_n_single():
    assert best_of_n(["only"], [1.0]) == "only"


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------


def test_normalize_scores_sum_to_1():
    scores = [1.0, 2.0, 3.0]
    normalized = normalize_scores(scores)
    assert abs(sum(normalized) - 1.0) < 1e-6


def test_normalize_scores_length():
    scores = [0.5, 1.5, 2.0, 0.1]
    normalized = normalize_scores(scores)
    assert len(normalized) == len(scores)


def test_normalize_scores_empty():
    assert normalize_scores([]) == []


# ---------------------------------------------------------------------------
# compute_response_diversity
# ---------------------------------------------------------------------------


def test_compute_response_diversity_all_same():
    # all same => 0.0
    result = compute_response_diversity(["X", "X", "X"])
    assert result == pytest.approx(0.0)


def test_compute_response_diversity_all_unique():
    result = compute_response_diversity(["A", "B", "C"])
    assert result == pytest.approx(1.0)


def test_compute_response_diversity_empty():
    assert compute_response_diversity([]) == 0.0


def test_compute_response_diversity_partial():
    # 2 unique out of 4: (2-1)/(4-1) = 1/3
    result = compute_response_diversity(["A", "A", "B", "B"])
    assert result == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# ComputeScaler
# ---------------------------------------------------------------------------


def test_compute_scaler_generate_n_length():
    cfg = ScalingConfig(n_samples=5)
    scaler = ComputeScaler(generate_fn=make_mock_generate(), config=cfg)
    responses = scaler.generate_n("test prompt")
    assert len(responses) == 5


def test_compute_scaler_aggregate_majority_vote():
    # Force all responses to be "A" by using a fixed generate_fn
    cfg = ScalingConfig(n_samples=4, aggregation="majority_vote")
    scaler = ComputeScaler(generate_fn=lambda p: "A", config=cfg)
    responses = ["A", "A", "B"]
    result = scaler.aggregate("prompt", responses)
    assert result == "A"


def test_compute_scaler_call_returns_tuple():
    cfg = ScalingConfig(n_samples=3, aggregation="majority_vote")
    scaler = ComputeScaler(generate_fn=make_mock_generate(seed=0), config=cfg)
    output = scaler("hello")
    assert isinstance(output, tuple)
    assert len(output) == 2


def test_compute_scaler_metadata_keys():
    cfg = ScalingConfig(n_samples=3, aggregation="majority_vote")
    scaler = ComputeScaler(generate_fn=make_mock_generate(seed=1), config=cfg)
    _, metadata = scaler("hello")
    for key in ("n_samples", "n_unique", "diversity", "aggregation"):
        assert key in metadata, f"Missing metadata key: {key}"


def test_compute_scaler_metadata_n_samples():
    cfg = ScalingConfig(n_samples=6, aggregation="majority_vote")
    scaler = ComputeScaler(generate_fn=make_mock_generate(seed=2), config=cfg)
    _, metadata = scaler("hello")
    assert metadata["n_samples"] == 6


def test_compute_scaler_best_of_n_aggregation():
    cfg = ScalingConfig(n_samples=3, aggregation="best_of_n")
    scores_map = {"A": 1.0, "B": 2.0, "C": 3.0}
    score_fn = lambda prompt, resp: scores_map.get(resp, 0.0)  # noqa: E731
    # Force a known sequence
    responses_iter = iter(["A", "B", "C"])
    scaler = ComputeScaler(
        generate_fn=lambda p: next(responses_iter),
        score_fn=score_fn,
        config=cfg,
    )
    best, _ = scaler("prompt")
    assert best == "C"


# ---------------------------------------------------------------------------
# PassAtK
# ---------------------------------------------------------------------------


def test_pass_at_k_n1_k1_c1():
    pak = PassAtK()
    result = pak.estimate(n_samples=1, k=1, c=1)
    assert result == pytest.approx(1.0)


def test_pass_at_k_n1_k1_c0():
    pak = PassAtK()
    result = pak.estimate(n_samples=1, k=1, c=0)
    assert result == pytest.approx(0.0)


def test_pass_at_k_n5_k1_c5():
    # All correct: should pass@1 = 1.0
    pak = PassAtK()
    result = pak.estimate(n_samples=5, k=1, c=5)
    assert result == pytest.approx(1.0)


def test_pass_at_k_n4_k2_c2():
    # 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
    pak = PassAtK()
    result = pak.estimate(n_samples=4, k=2, c=2)
    assert result == pytest.approx(1 - 1 / 6)


def test_compute_from_results_all_true():
    pak = PassAtK()
    results = [True, True, True, True]
    result = pak.compute_from_results(results, k=1)
    assert result == pytest.approx(1.0)


def test_compute_from_results_all_false():
    pak = PassAtK()
    results = [False, False, False]
    result = pak.compute_from_results(results, k=1)
    assert result == pytest.approx(0.0)


def test_compute_from_results_mixed():
    pak = PassAtK()
    # 2 correct out of 4, k=1: 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
    results = [True, False, True, False]
    result = pak.compute_from_results(results, k=1)
    assert result == pytest.approx(0.5)
