"""Integration test for src/data/synthetic_math.py.

Verifies end-to-end generation of 30 mixed problems, correct-answer verification,
difficulty_stats totals, and registry wiring.
"""

from __future__ import annotations

from src.data.synthetic_math import (
    DATA_REGISTRY,
    SyntheticMathConfig,
    SyntheticMathGenerator,
)


def test_registry_wired():
    """SyntheticMathGenerator must be accessible via DATA_REGISTRY."""
    assert "synthetic_math" in DATA_REGISTRY
    assert DATA_REGISTRY["synthetic_math"] is SyntheticMathGenerator


def test_integration_generate_30_mixed():
    """Generate 30 mixed problems and verify all are well-formed."""
    cfg = SyntheticMathConfig(seed=131)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(30)

    assert len(problems) == 30

    valid_types = {"arithmetic", "algebra", "combinatorics"}
    valid_difficulties = {"easy", "medium", "hard"}

    for p in problems:
        assert p.problem_id != "", f"Empty problem_id: {p}"
        assert len(p.problem_text) > 0, f"Empty problem_text: {p.problem_id}"
        assert len(p.answer) > 0, f"Empty answer: {p.problem_id}"
        assert p.problem_type in valid_types, f"Bad type: {p.problem_type}"
        assert p.difficulty in valid_difficulties, f"Bad difficulty: {p.difficulty}"
        assert p.answer_numeric is None or isinstance(p.answer_numeric, float)


def test_integration_verify_10_correct_answers():
    """verify() must return True for the first 10 problems' own answers."""
    cfg = SyntheticMathConfig(seed=131)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(30)

    for p in problems[:10]:
        result = gen.verify(p, p.answer)
        assert result is True, (
            f"verify() returned False for correct answer '{p.answer}' on problem {p.problem_id}"
        )


def test_integration_difficulty_stats_totals():
    """difficulty_stats totals must match number of generated problems."""
    cfg = SyntheticMathConfig(seed=131)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(30)
    stats = gen.difficulty_stats(problems)

    assert stats["total"] == 30
    assert sum(stats["by_difficulty"].values()) == 30
    assert sum(stats["by_type"].values()) == 30


def test_integration_registry_instantiable():
    """Generator obtained from registry should produce problems normally."""
    GenClass = DATA_REGISTRY["synthetic_math"]
    gen = GenClass(SyntheticMathConfig(seed=42))
    problems = gen.generate(5)
    assert len(problems) == 5
    for p in problems:
        assert gen.verify(p, p.answer)


def test_integration_mixed_type_coverage():
    """With enough problems, all three types should appear."""
    cfg = SyntheticMathConfig(seed=7)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(60)
    stats = gen.difficulty_stats(problems)
    found_types = set(stats["by_type"].keys())
    assert found_types == {"arithmetic", "algebra", "combinatorics"}, (
        f"Not all problem types appeared: {found_types}"
    )


def test_integration_all_problem_ids_unique():
    """All 30 problems must have distinct IDs."""
    cfg = SyntheticMathConfig(seed=131)
    gen = SyntheticMathGenerator(cfg)
    problems = gen.generate(30)
    ids = [p.problem_id for p in problems]
    assert len(ids) == len(set(ids)), "Duplicate problem IDs found"
