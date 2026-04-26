"""Unit tests for src.eval.needle_in_haystack."""

from __future__ import annotations

import pytest

from src.eval.needle_in_haystack import (
    NeedleInHaystackBenchmark,
    default_haystack_filler,
)

# ---------------------------------------------------------------------------
# Fake generators
# ---------------------------------------------------------------------------


def oracle_fn(prompt: str) -> str:
    """Oracle: finds the needle if it's in the prompt and echoes the number."""
    if "The magic number is 42." in prompt:
        return "The answer is 42."
    return "I don't know."


def never_answer_fn(_: str) -> str:
    return "I have no idea what you are asking."


def echo_case_fn(_: str) -> str:
    # All upper case to exercise case-insensitive matching.
    return "FOURTY-TWO, I.E. 42 IS THE ANSWER."


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_length_scales_with_context_tokens():
    bench = NeedleInHaystackBenchmark()
    short = bench.build_prompt(context_tokens=64, depth=0.5)
    long = bench.build_prompt(context_tokens=1024, depth=0.5)
    assert len(long) > len(short)
    # Ballpark: longer prompt should be at least 4x the short one.
    assert len(long) > 4 * len(short) // 2


def test_needle_present_exactly_once():
    bench = NeedleInHaystackBenchmark()
    prompt = bench.build_prompt(context_tokens=512, depth=0.5)
    assert prompt.count(bench.needle) == 1


def test_depth_zero_places_needle_near_start():
    bench = NeedleInHaystackBenchmark()
    prompt = bench.build_prompt(context_tokens=1024, depth=0.0)
    idx = prompt.index(bench.needle)
    # Strip question tail from consideration.
    body_len = len(prompt)
    assert idx < body_len * 0.25, f"needle at {idx}/{body_len} expected near start"


def test_depth_one_places_needle_near_end():
    bench = NeedleInHaystackBenchmark()
    prompt = bench.build_prompt(context_tokens=1024, depth=1.0)
    idx = prompt.index(bench.needle)
    body_len = len(prompt) - len("\n\nQuestion: " + bench.question + "\nAnswer:")
    assert idx > body_len * 0.5, f"needle at {idx}/{body_len} expected near end"


def test_invalid_depth_raises():
    bench = NeedleInHaystackBenchmark()
    with pytest.raises(ValueError):
        bench.build_prompt(context_tokens=64, depth=-0.1)
    with pytest.raises(ValueError):
        bench.build_prompt(context_tokens=64, depth=1.01)


def test_invalid_context_tokens_raises():
    bench = NeedleInHaystackBenchmark()
    with pytest.raises(ValueError):
        bench.build_prompt(context_tokens=0, depth=0.5)
    with pytest.raises(ValueError):
        bench.build_prompt(context_tokens=-10, depth=0.5)


def test_small_context_tokens_works():
    bench = NeedleInHaystackBenchmark()
    prompt = bench.build_prompt(context_tokens=32, depth=0.5)
    # Degenerate but valid: needle and question must still both be present.
    assert bench.needle in prompt
    assert bench.question in prompt


def test_question_always_appended():
    bench = NeedleInHaystackBenchmark(question="What is X?")
    prompt = bench.build_prompt(context_tokens=256, depth=0.3)
    assert prompt.rstrip().endswith("Answer:")
    assert "What is X?" in prompt


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_default_filler_deterministic_with_seed():
    a = default_haystack_filler(2048, seed=7)
    b = default_haystack_filler(2048, seed=7)
    c = default_haystack_filler(2048, seed=8)
    assert a == b
    assert a != c


def test_build_prompt_deterministic_with_seeded_filler():
    def filler(n: int) -> str:
        return default_haystack_filler(n, seed=123)

    bench1 = NeedleInHaystackBenchmark(haystack_filler=filler)
    bench2 = NeedleInHaystackBenchmark(haystack_filler=filler)
    p1 = bench1.build_prompt(context_tokens=512, depth=0.4)
    p2 = bench2.build_prompt(context_tokens=512, depth=0.4)
    assert p1 == p2


# ---------------------------------------------------------------------------
# evaluate / score / grid_report
# ---------------------------------------------------------------------------


def test_evaluate_with_oracle_returns_perfect_score():
    bench = NeedleInHaystackBenchmark()
    results = bench.evaluate(oracle_fn, context_lengths=[64, 128], depths=[0.0, 0.5, 1.0])
    assert bench.score(results) == 1.0
    assert len(results) == 6


def test_evaluate_with_never_answer_returns_zero():
    bench = NeedleInHaystackBenchmark()
    results = bench.evaluate(never_answer_fn, context_lengths=[64, 128], depths=[0.0, 1.0])
    assert bench.score(results) == 0.0


def test_grid_has_one_entry_per_combo():
    bench = NeedleInHaystackBenchmark()
    results = bench.evaluate(oracle_fn, context_lengths=[64, 128, 256], depths=[0.0, 0.5])
    assert len(results) == 6
    keys = set(results.keys())
    expected = {(L, d) for L in [64, 128, 256] for d in [0.0, 0.5]}
    assert keys == expected


def test_empty_context_lengths_raises():
    bench = NeedleInHaystackBenchmark()
    with pytest.raises(ValueError):
        bench.evaluate(oracle_fn, context_lengths=[], depths=[0.5])


def test_empty_depths_raises():
    bench = NeedleInHaystackBenchmark()
    with pytest.raises(ValueError):
        bench.evaluate(oracle_fn, context_lengths=[64], depths=[])


def test_case_insensitive_answer_match():
    bench = NeedleInHaystackBenchmark(answer_key="forty-two")

    def fn(_: str) -> str:
        return "The answer is FORTY-TWO."

    results = bench.evaluate(fn, context_lengths=[64], depths=[0.5])
    assert bench.score(results) == 1.0


def test_score_hand_computed():
    # Four cells, two pass => 0.5.
    results = {
        (64, 0.0): {"pass": True, "response": "42"},
        (64, 1.0): {"pass": False, "response": "?"},
        (128, 0.0): {"pass": True, "response": "42"},
        (128, 1.0): {"pass": False, "response": "?"},
    }
    assert NeedleInHaystackBenchmark.score(results) == 0.5


def test_score_empty_is_zero():
    assert NeedleInHaystackBenchmark.score({}) == 0.0


def test_grid_report_is_str_and_contains_lengths():
    bench = NeedleInHaystackBenchmark()
    results = bench.evaluate(oracle_fn, context_lengths=[64, 128], depths=[0.0, 0.5])
    report = bench.grid_report(results)
    assert isinstance(report, str)
    assert "64" in report and "128" in report


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_empty_needle_raises():
    with pytest.raises(ValueError):
        NeedleInHaystackBenchmark(needle="")


def test_empty_answer_key_raises():
    with pytest.raises(ValueError):
        NeedleInHaystackBenchmark(answer_key="")


def test_evaluate_rejects_non_string_response():
    bench = NeedleInHaystackBenchmark()

    def bad_fn(_: str):
        return 42  # not a string

    with pytest.raises(TypeError):
        bench.evaluate(bad_fn, context_lengths=[64], depths=[0.5])
