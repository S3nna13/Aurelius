"""Unit tests for src/data/synthetic_math.py.

Tests 1-15 cover: config defaults, generation count, unique IDs, per-type
correctness, difficulty distribution, verify correctness, numeric matching,
wrong-answer rejection, answer_numeric typing, easy-problem magnitude,
determinism, difficulty_stats shape, and difficulty_stats counts.
"""

from __future__ import annotations

from src.data.synthetic_math import (
    MathProblem,
    SyntheticMathConfig,
    SyntheticMathGenerator,
)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = SyntheticMathConfig()
    assert cfg.seed == 42
    assert len(cfg.problem_types) == 3
    assert set(cfg.problem_types) == {"arithmetic", "algebra", "combinatorics"}
    assert set(cfg.difficulty_distribution) == {"easy", "medium", "hard"}
    assert abs(sum(cfg.difficulty_distribution.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. test_generate_count
# ---------------------------------------------------------------------------
def test_generate_count():
    gen = SyntheticMathGenerator()
    problems = gen.generate(10)
    assert len(problems) == 10


# ---------------------------------------------------------------------------
# 3. test_generate_unique_ids
# ---------------------------------------------------------------------------
def test_generate_unique_ids():
    gen = SyntheticMathGenerator()
    problems = gen.generate(50)
    ids = [p.problem_id for p in problems]
    assert len(ids) == len(set(ids)), "Problem IDs are not all unique"


# ---------------------------------------------------------------------------
# 4. test_generate_arithmetic
# ---------------------------------------------------------------------------
def test_generate_arithmetic():
    gen = SyntheticMathGenerator()
    problems = gen.generate(20, problem_type="arithmetic")
    assert all(p.problem_type == "arithmetic" for p in problems)
    assert all(isinstance(p.problem_text, str) and len(p.problem_text) > 0 for p in problems)


# ---------------------------------------------------------------------------
# 5. test_generate_algebra
# ---------------------------------------------------------------------------
def test_generate_algebra():
    SyntheticMathGenerator()
    # Easy algebra: ax = b, answer should be a positive integer
    cfg = SyntheticMathConfig(seed=42)
    gen2 = SyntheticMathGenerator(cfg)
    problems = gen2.generate(30, problem_type="algebra")
    algebra_problems = [p for p in problems if p.difficulty == "easy"]
    for p in algebra_problems:
        # answer_numeric should be set and be a float
        assert p.answer_numeric is not None
        assert isinstance(p.answer_numeric, float)
        # answer should parse as an integer
        assert p.answer == str(int(float(p.answer)))


# ---------------------------------------------------------------------------
# 6. test_generate_combinatorics
# ---------------------------------------------------------------------------
def test_generate_combinatorics():
    gen = SyntheticMathGenerator()
    problems = gen.generate(30, problem_type="combinatorics")
    easy_problems = [p for p in problems if p.difficulty == "easy"]
    assert len(easy_problems) > 0, "Expected at least one easy combinatorics problem"
    for p in easy_problems:
        # n! is always a positive integer
        val = int(p.answer)
        assert val > 0
        # sanity: n! >= 6 for n >= 3
        assert val >= 6


# ---------------------------------------------------------------------------
# 7. test_difficulty_distribution
# ---------------------------------------------------------------------------
def test_difficulty_distribution():
    """Over 200 samples the observed distribution should be within 15 pp of target."""
    cfg = SyntheticMathConfig(seed=7)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(200)
    stats = gen.difficulty_stats(problems)
    total = stats["total"]
    for diff, expected_ratio in cfg.difficulty_distribution.items():
        count = stats["by_difficulty"].get(diff, 0)
        observed = count / total
        assert abs(observed - expected_ratio) < 0.15, (
            f"Difficulty '{diff}': expected ~{expected_ratio:.2f}, got {observed:.2f}"
        )


# ---------------------------------------------------------------------------
# 8. test_verify_correct
# ---------------------------------------------------------------------------
def test_verify_correct():
    gen = SyntheticMathGenerator()
    problems = gen.generate(20)
    for p in problems:
        assert gen.verify(p, p.answer), (
            f"verify() returned False for correct answer on {p.problem_id}"
        )


# ---------------------------------------------------------------------------
# 9. test_verify_numeric_match
# ---------------------------------------------------------------------------
def test_verify_numeric_match():
    gen = SyntheticMathGenerator()
    # Build a problem whose answer is "42"
    p = MathProblem(
        problem_id="test_0",
        problem_text="What is 6 × 7?",
        answer="42",
        answer_numeric=42.0,
        difficulty="easy",
        problem_type="arithmetic",
    )
    assert gen.verify(p, "42.0"), "42.0 should match answer '42' numerically"
    assert gen.verify(p, "42"), "42 should match answer '42'"


# ---------------------------------------------------------------------------
# 10. test_verify_wrong
# ---------------------------------------------------------------------------
def test_verify_wrong():
    gen = SyntheticMathGenerator()
    p = MathProblem(
        problem_id="test_1",
        problem_text="What is 2 + 2?",
        answer="4",
        answer_numeric=4.0,
        difficulty="easy",
        problem_type="arithmetic",
    )
    assert not gen.verify(p, "5"), "verify() should return False for wrong answer"
    assert not gen.verify(p, ""), "verify() should return False for empty string"
    assert not gen.verify(p, "four"), "verify() should return False for word answer"


# ---------------------------------------------------------------------------
# 11. test_answer_numeric_set
# ---------------------------------------------------------------------------
def test_answer_numeric_set():
    gen = SyntheticMathGenerator()
    problems = gen.generate(30)
    for p in problems:
        assert p.answer_numeric is None or isinstance(p.answer_numeric, float), (
            f"answer_numeric must be float or None, got {type(p.answer_numeric)} for {p.problem_id}"
        )


# ---------------------------------------------------------------------------
# 12. test_difficulty_easy_small_numbers
# ---------------------------------------------------------------------------
def test_difficulty_easy_small_numbers():
    """Easy arithmetic problems should involve smaller numbers than hard ones."""
    cfg = SyntheticMathConfig(seed=42, max_number=1000)
    gen = SyntheticMathGenerator(cfg)
    easy = gen.generate(50, problem_type="arithmetic")
    easy_problems = [p for p in easy if p.difficulty == "easy"]
    # For easy problems, answer_numeric should be a manageable number
    # max easy cap is max_number/10 = 100; max product is 100*100 = 10000
    for p in easy_problems:
        if p.answer_numeric is not None:
            assert abs(p.answer_numeric) <= 10_000, (
                f"Easy problem answer {p.answer_numeric} seems too large"
            )


# ---------------------------------------------------------------------------
# 13. test_determinism
# ---------------------------------------------------------------------------
def test_determinism():
    """Same seed produces identical problems."""
    cfg1 = SyntheticMathConfig(seed=99)
    cfg2 = SyntheticMathConfig(seed=99)
    gen1 = SyntheticMathGenerator(cfg1)
    gen2 = SyntheticMathGenerator(cfg2)
    p1 = gen1.generate(20)
    p2 = gen2.generate(20)
    for a, b in zip(p1, p2):
        assert a.problem_id == b.problem_id
        assert a.problem_text == b.problem_text
        assert a.answer == b.answer


# ---------------------------------------------------------------------------
# 14. test_difficulty_stats_keys
# ---------------------------------------------------------------------------
def test_difficulty_stats_keys():
    gen = SyntheticMathGenerator()
    problems = gen.generate(10)
    stats = gen.difficulty_stats(problems)
    assert "total" in stats
    assert "by_difficulty" in stats
    assert "by_type" in stats
    assert isinstance(stats["by_difficulty"], dict)
    assert isinstance(stats["by_type"], dict)


# ---------------------------------------------------------------------------
# 15. test_difficulty_stats_counts
# ---------------------------------------------------------------------------
def test_difficulty_stats_counts():
    gen = SyntheticMathGenerator()
    problems = gen.generate(50)
    stats = gen.difficulty_stats(problems)
    assert stats["total"] == 50
    assert sum(stats["by_difficulty"].values()) == stats["total"]
    assert sum(stats["by_type"].values()) == stats["total"]
