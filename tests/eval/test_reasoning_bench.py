"""Tests for synthetic reasoning benchmarks."""

from __future__ import annotations

import random
import pytest
import torch

from src.eval.reasoning_bench import (
    BenchmarkConfig,
    Problem,
    ReasoningBenchmark,
    generate_arithmetic_problem,
    generate_analogy_problem,
    generate_logic_problem,
    generate_sequence_problem,
    evaluate_chain_of_thought,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _encode(text: str) -> list[int]:
    """Trivial byte-level tokenizer (vocab_size=256)."""
    return [b % 256 for b in text.encode("utf-8")][:128]  # cap length


def _decode(ids: list[int]) -> str:
    """Decode byte tokens back to string (best-effort)."""
    return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture
def fast_config():
    return BenchmarkConfig(
        n_problems=4,
        difficulty="medium",
        problem_types=["arithmetic", "logic", "analogy", "sequence"],
        seed=42,
    )


@pytest.fixture
def bench(fast_config):
    return ReasoningBenchmark(fast_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_benchmark_config_defaults():
    """BenchmarkConfig has expected default values."""
    cfg = BenchmarkConfig()
    assert cfg.n_problems == 100
    assert cfg.difficulty == "medium"
    assert set(cfg.problem_types) == {"arithmetic", "logic", "analogy", "sequence"}
    assert cfg.seed == 42
    assert cfg.max_answer_len == 32


def test_problem_fields():
    """Problem dataclass exposes all required fields."""
    p = Problem(
        type="arithmetic",
        prompt="What is 2 + 3?",
        answer="5",
        difficulty="easy",
        metadata={"a": 2, "b": 3},
    )
    assert p.type == "arithmetic"
    assert p.prompt == "What is 2 + 3?"
    assert p.answer == "5"
    assert p.difficulty == "easy"
    assert isinstance(p.metadata, dict)


def test_generate_arithmetic_easy():
    """Easy arithmetic problem has a numeric string answer."""
    rng = random.Random(0)
    problem = generate_arithmetic_problem("easy", rng)
    assert problem.type == "arithmetic"
    assert problem.difficulty == "easy"
    # Answer must be a numeric string
    int(problem.answer)  # raises ValueError if not numeric


def test_generate_arithmetic_medium():
    """Medium arithmetic problem has valid integer answer."""
    rng = random.Random(1)
    problem = generate_arithmetic_problem("medium", rng)
    assert problem.type == "arithmetic"
    result = int(problem.answer)
    # Verify: a * b + c
    meta = problem.metadata
    assert result == meta["a"] * meta["b"] + meta["c"]


def test_generate_logic_problem():
    """Logic problem answer is one of Yes/No/True/False."""
    rng = random.Random(0)
    valid_answers = {"Yes", "No", "True", "False"}
    for difficulty in ("easy", "medium", "hard"):
        p = generate_logic_problem(difficulty, rng)
        assert p.type == "logic"
        assert p.answer in valid_answers, f"Unexpected answer {p.answer!r} for {difficulty}"


def test_generate_analogy_problem_has_choices():
    """Analogy problem metadata contains a non-empty choices list."""
    rng = random.Random(7)
    problem = generate_analogy_problem("medium", rng)
    assert problem.type == "analogy"
    assert "choices" in problem.metadata
    choices = problem.metadata["choices"]
    assert isinstance(choices, list)
    assert len(choices) > 0
    # Correct answer should be in choices
    assert problem.answer in choices


def test_generate_sequence_problem_easy():
    """Easy sequence problem generates correct arithmetic next-value."""
    rng = random.Random(99)
    problem = generate_sequence_problem("easy", rng)
    assert problem.type == "sequence"
    assert problem.difficulty == "easy"
    meta = problem.metadata
    start, step = meta["start"], meta["step"]
    seq = meta["sequence"]
    # Next value after 4 elements
    expected = start + 4 * step
    assert problem.answer == str(expected)
    # Sequence is arithmetic with correct step
    for i in range(1, len(seq)):
        assert seq[i] - seq[i - 1] == step


def test_reasoning_benchmark_problem_count(fast_config):
    """ReasoningBenchmark generates exactly n_problems problems."""
    bench = ReasoningBenchmark(fast_config)
    assert len(bench.problems) == fast_config.n_problems


def test_reasoning_benchmark_types(fast_config):
    """ReasoningBenchmark generates all requested problem types."""
    bench = ReasoningBenchmark(fast_config)
    generated_types = {p.type for p in bench.problems}
    assert generated_types == set(fast_config.problem_types)


def test_reasoning_benchmark_evaluate_model_keys(small_model, bench):
    """evaluate_model returns dict with overall_accuracy, by_type, by_difficulty."""
    results = bench.evaluate_model(
        model=small_model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        max_new_tokens=4,
    )
    assert "overall_accuracy" in results
    assert "by_type" in results
    assert "by_difficulty" in results
    assert isinstance(results["overall_accuracy"], float)
    assert 0.0 <= results["overall_accuracy"] <= 1.0
    assert isinstance(results["by_type"], dict)
    assert isinstance(results["by_difficulty"], dict)


def test_format_few_shot_longer(bench):
    """Few-shot prompt is longer than zero-shot prompt."""
    problem = bench.problems[0]
    zero_shot = problem.prompt
    few_shot = bench.format_few_shot(problem, n_shots=3)
    assert len(few_shot) > len(zero_shot)
    # The original prompt should appear in the few-shot prompt
    assert problem.prompt in few_shot


def test_evaluate_chain_of_thought_keys(small_model, bench):
    """evaluate_chain_of_thought returns cot_accuracy, direct_accuracy, cot_improvement."""
    results = evaluate_chain_of_thought(
        model=small_model,
        problems=bench.problems,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        max_new_tokens=4,
    )
    assert "cot_accuracy" in results
    assert "direct_accuracy" in results
    assert "cot_improvement" in results
    assert isinstance(results["cot_accuracy"], float)
    assert isinstance(results["direct_accuracy"], float)
    # cot_improvement = cot_accuracy - direct_accuracy
    assert abs(results["cot_improvement"] - (results["cot_accuracy"] - results["direct_accuracy"])) < 1e-9
