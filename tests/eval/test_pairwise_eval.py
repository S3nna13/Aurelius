"""Tests for AlpacaEval-style pairwise evaluation module."""

from __future__ import annotations

import pytest

from src.eval.pairwise_eval import (
    AlpacaEvalPipeline,
    PairwiseAnnotator,
    PairwiseConfig,
    PairwiseResult,
    build_pairwise_prompt,
    compute_win_rate,
    extract_pairwise_winner,
)

# ---------------------------------------------------------------------------
# Mock generate_fn — always returns "Winner: A"
# ---------------------------------------------------------------------------


def _mock_generate(prompt: str) -> str:  # noqa: ARG001
    return "The response was good overall.\nWinner: A"


# ---------------------------------------------------------------------------
# build_pairwise_prompt
# ---------------------------------------------------------------------------


def test_build_pairwise_prompt_contains_instruction():
    prompt = build_pairwise_prompt("What is gravity?", "It pulls things.", "A force.")
    assert "What is gravity?" in prompt


def test_build_pairwise_prompt_contains_response_a():
    prompt = build_pairwise_prompt("Q", "Answer A", "Answer B")
    assert "Response A" in prompt


def test_build_pairwise_prompt_contains_response_b():
    prompt = build_pairwise_prompt("Q", "Answer A", "Answer B")
    assert "Response B" in prompt


# ---------------------------------------------------------------------------
# extract_pairwise_winner
# ---------------------------------------------------------------------------


def test_extract_winner_a():
    assert extract_pairwise_winner("Winner: A") == "A"


def test_extract_winner_b():
    assert extract_pairwise_winner("Winner: B") == "B"


def test_extract_winner_tie():
    assert extract_pairwise_winner("Winner: tie") == "tie"


def test_extract_winner_garbage_defaults_tie():
    assert extract_pairwise_winner("completely unparseable output %%%") == "tie"


# ---------------------------------------------------------------------------
# PairwiseResult dataclass
# ---------------------------------------------------------------------------


def test_pairwise_result_creation():
    result = PairwiseResult(
        instruction="Q",
        output_a="A",
        output_b="B",
        winner="A",
    )
    assert result.instruction == "Q"
    assert result.output_a == "A"
    assert result.output_b == "B"
    assert result.winner == "A"
    assert result.judge_reasoning == ""
    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# PairwiseConfig defaults
# ---------------------------------------------------------------------------


def test_pairwise_config_defaults():
    cfg = PairwiseConfig()
    assert cfg.n_annotators == 1
    assert cfg.temperature == 0.0
    assert cfg.reference_model == "reference"
    assert cfg.annotator_model == "judge"


# ---------------------------------------------------------------------------
# PairwiseAnnotator.annotate
# ---------------------------------------------------------------------------


def test_annotate_returns_pairwise_result():
    annotator = PairwiseAnnotator(_mock_generate, PairwiseConfig())
    result = annotator.annotate("What is AI?", "AI is cool.", "AI is intelligent.")
    assert isinstance(result, PairwiseResult)


def test_annotate_winner_valid_value():
    annotator = PairwiseAnnotator(_mock_generate, PairwiseConfig())
    result = annotator.annotate("Q", "A", "B")
    assert result.winner in ("A", "B", "tie")


# ---------------------------------------------------------------------------
# PairwiseAnnotator.annotate_batch
# ---------------------------------------------------------------------------


def test_annotate_batch_length_matches_input():
    annotator = PairwiseAnnotator(_mock_generate, PairwiseConfig())
    items = [
        ("Q1", "A1", "B1"),
        ("Q2", "A2", "B2"),
        ("Q3", "A3", "B3"),
    ]
    results = annotator.annotate_batch(items)
    assert len(results) == len(items)


def test_annotate_batch_returns_list_of_pairwise_result():
    annotator = PairwiseAnnotator(_mock_generate, PairwiseConfig())
    items = [("Q", "A", "B"), ("Q2", "A2", "B2")]
    results = annotator.annotate_batch(items)
    assert all(isinstance(r, PairwiseResult) for r in results)


# ---------------------------------------------------------------------------
# compute_win_rate
# ---------------------------------------------------------------------------


def test_compute_win_rate_has_required_keys():
    results = [PairwiseResult("Q", "A", "B", "A")]
    metrics = compute_win_rate(results)
    for key in ("win_rate", "loss_rate", "tie_rate", "n_comparisons"):
        assert key in metrics


def test_compute_win_rate_rates_sum_to_one():
    results = [
        PairwiseResult("Q1", "A1", "B1", "A"),
        PairwiseResult("Q2", "A2", "B2", "B"),
        PairwiseResult("Q3", "A3", "B3", "tie"),
    ]
    metrics = compute_win_rate(results)
    total = metrics["win_rate"] + metrics["loss_rate"] + metrics["tie_rate"]
    assert total == pytest.approx(1.0)


def test_compute_win_rate_all_a_wins():
    results = [PairwiseResult("Q", "A", "B", "A") for _ in range(5)]
    metrics = compute_win_rate(results)
    assert metrics["win_rate"] == pytest.approx(1.0)


def test_compute_win_rate_empty_list():
    metrics = compute_win_rate([])
    assert metrics["win_rate"] == 0.0
    assert metrics["loss_rate"] == 0.0
    assert metrics["tie_rate"] == 0.0
    assert metrics["n_comparisons"] == 0


# ---------------------------------------------------------------------------
# AlpacaEvalPipeline.evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict_with_win_rate():
    pipeline = AlpacaEvalPipeline(PairwiseAnnotator(_mock_generate, PairwiseConfig()))
    result = pipeline.evaluate(["Q1", "Q2"], ["A1", "A2"], ["B1", "B2"])
    assert "win_rate" in result


def test_evaluate_instruction_count_matches_outputs():
    pipeline = AlpacaEvalPipeline(PairwiseAnnotator(_mock_generate, PairwiseConfig()))
    instructions = ["Q1", "Q2", "Q3"]
    model_outputs = ["A1", "A2", "A3"]
    reference_outputs = ["B1", "B2", "B3"]
    result = pipeline.evaluate(instructions, model_outputs, reference_outputs)
    assert result["n_comparisons"] == len(instructions)


# ---------------------------------------------------------------------------
# AlpacaEvalPipeline.bootstrap_confidence_interval
# ---------------------------------------------------------------------------


def test_bootstrap_confidence_interval_returns_two_floats():
    pipeline = AlpacaEvalPipeline(PairwiseAnnotator(_mock_generate, PairwiseConfig()))
    results = [PairwiseResult("Q", "A", "B", "A") for _ in range(10)]
    ci = pipeline.bootstrap_confidence_interval(results, n_bootstrap=50)
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert isinstance(ci[0], float)
    assert isinstance(ci[1], float)


def test_bootstrap_lower_le_upper():
    pipeline = AlpacaEvalPipeline(PairwiseAnnotator(_mock_generate, PairwiseConfig()))
    results = [
        PairwiseResult("Q1", "A1", "B1", "A"),
        PairwiseResult("Q2", "A2", "B2", "B"),
        PairwiseResult("Q3", "A3", "B3", "tie"),
        PairwiseResult("Q4", "A4", "B4", "A"),
        PairwiseResult("Q5", "A5", "B5", "B"),
    ]
    lower, upper = pipeline.bootstrap_confidence_interval(results, n_bootstrap=200)
    assert lower <= upper
