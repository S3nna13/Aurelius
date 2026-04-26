"""Tests for symbolic reasoning evaluation."""

from __future__ import annotations

import random

import pytest
import torch

from src.eval.symbolic_reasoning import (
    LogicProblem,
    SymbolicReasoningBenchmark,
    evaluate_answer,
    generate_arithmetic,
    generate_boolean_logic,
    generate_syllogism,
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
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _encode(text: str) -> list[int]:
    """Trivial byte-level tokenizer (vocab_size=256)."""
    return [b % 256 for b in text.encode("utf-8")][:128]


def _decode(ids: list[int]) -> str:
    """Decode byte tokens back to string (best-effort)."""
    return bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Tests: LogicProblem
# ---------------------------------------------------------------------------


def test_logic_problem_fields_and_defaults():
    """LogicProblem has all required fields and difficulty defaults to 1."""
    p = LogicProblem(
        problem_type="syllogism",
        premises=["All A are B.", "All B are C."],
        question="Are all A C?",
        answer="yes",
    )
    assert p.problem_type == "syllogism"
    assert p.premises == ["All A are B.", "All B are C."]
    assert p.question == "Are all A C?"
    assert p.answer == "yes"
    assert p.difficulty == 1


# ---------------------------------------------------------------------------
# Tests: generate_syllogism
# ---------------------------------------------------------------------------


def test_generate_syllogism_returns_logic_problem():
    """generate_syllogism returns a LogicProblem with type='syllogism'."""
    rng = random.Random(0)
    p = generate_syllogism(rng)
    assert isinstance(p, LogicProblem)
    assert p.problem_type == "syllogism"


def test_generate_syllogism_premises_non_empty():
    """generate_syllogism produces a non-empty premises list."""
    rng = random.Random(1)
    p = generate_syllogism(rng)
    assert isinstance(p.premises, list)
    assert len(p.premises) > 0


def test_generate_syllogism_answer_yes_or_no():
    """generate_syllogism answer is 'yes' or 'no'."""
    rng = random.Random(2)
    for _ in range(10):
        p = generate_syllogism(rng)
        assert p.answer.lower() in ("yes", "no"), f"Unexpected answer: {p.answer!r}"


# ---------------------------------------------------------------------------
# Tests: generate_arithmetic
# ---------------------------------------------------------------------------


def test_generate_arithmetic_returns_logic_problem():
    """generate_arithmetic returns a LogicProblem."""
    rng = random.Random(3)
    p = generate_arithmetic(rng)
    assert isinstance(p, LogicProblem)
    assert p.problem_type == "arithmetic"


def test_generate_arithmetic_answer_is_numeric_string():
    """generate_arithmetic answer is a numeric string."""
    rng = random.Random(4)
    p = generate_arithmetic(rng)
    int(p.answer)  # raises ValueError if not numeric


# ---------------------------------------------------------------------------
# Tests: generate_boolean_logic
# ---------------------------------------------------------------------------


def test_generate_boolean_logic_returns_logic_problem():
    """generate_boolean_logic returns a LogicProblem with type='boolean'."""
    rng = random.Random(5)
    p = generate_boolean_logic(rng)
    assert isinstance(p, LogicProblem)
    assert p.problem_type == "boolean"


def test_generate_boolean_logic_answer_true_or_false():
    """generate_boolean_logic answer is 'True' or 'False'."""
    rng = random.Random(6)
    for _ in range(10):
        p = generate_boolean_logic(rng)
        assert p.answer in ("True", "False"), f"Unexpected answer: {p.answer!r}"


# ---------------------------------------------------------------------------
# Tests: evaluate_answer
# ---------------------------------------------------------------------------


def test_evaluate_answer_exact_match():
    """Exact string match (after strip/lower) returns 1.0."""
    assert evaluate_answer("yes", "yes", "syllogism") == 1.0
    assert evaluate_answer("  Yes  ", "yes", "syllogism") == 1.0


def test_evaluate_answer_wrong_answer():
    """Wrong answer returns 0.0."""
    assert evaluate_answer("no", "yes", "syllogism") == 0.0
    assert evaluate_answer("banana", "yes", "syllogism") == 0.0


def test_evaluate_answer_arithmetic_numeric_match():
    """Arithmetic: '6' vs '6' returns 1.0."""
    assert evaluate_answer("6", "6", "arithmetic") == 1.0


def test_evaluate_answer_arithmetic_wrong():
    """Arithmetic: different integers return 0.0."""
    assert evaluate_answer("5", "6", "arithmetic") == 0.0


def test_evaluate_answer_boolean_case_insensitive():
    """Boolean: 'true' and 'True' are equivalent; 'yes' and 'True' are equivalent."""
    assert evaluate_answer("true", "True", "boolean") == 1.0
    assert evaluate_answer("yes", "True", "boolean") == 1.0
    assert evaluate_answer("false", "False", "boolean") == 1.0
    assert evaluate_answer("no", "False", "boolean") == 1.0


# ---------------------------------------------------------------------------
# Tests: SymbolicReasoningBenchmark
# ---------------------------------------------------------------------------


def test_benchmark_generates_correct_n_problems():
    """SymbolicReasoningBenchmark generates exactly n_problems problems."""
    bench = SymbolicReasoningBenchmark(n_problems=12, seed=42)
    assert len(bench.problems) == 12


def test_benchmark_evaluate_model_returns_required_keys(small_model):
    """evaluate_model returns 'accuracy', 'by_type', and 'n_problems' keys."""
    bench = SymbolicReasoningBenchmark(n_problems=6, seed=0)
    results = bench.evaluate_model(
        model=small_model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        max_new_tokens=4,
    )
    assert "accuracy" in results
    assert "by_type" in results
    assert "n_problems" in results
    assert isinstance(results["accuracy"], float)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert isinstance(results["by_type"], dict)
    assert results["n_problems"] == 6


def test_generate_prompt_contains_question():
    """generate_prompt output contains the problem question."""
    p = LogicProblem(
        problem_type="syllogism",
        premises=["All foo are bar."],
        question="Are all foo bar?",
        answer="yes",
    )
    prompt = SymbolicReasoningBenchmark.generate_prompt(p)
    assert "Are all foo bar?" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
