"""Tests for 15-dimension layered constitution scoring."""

from __future__ import annotations

import warnings

import pytest

from src.alignment.constitution_dimensions import (
    CONSTITUTION_DIMENSIONS,
    DEFAULT_CONSTITUTION_TEXT,
    DEFAULT_GRADERS,
    ConstitutionDimension,
    ConstitutionLevel,
    ConstitutionReport,
    ConstitutionScorer,
    DimensionGrade,
    heuristic_helpfulness_grader,
    heuristic_honesty_grader,
    heuristic_safety_grader,
)


def test_all_15_dimensions_loaded():
    assert len(CONSTITUTION_DIMENSIONS) == 15


def test_dimension_ids_unique():
    ids = [d.id for d in CONSTITUTION_DIMENSIONS]
    assert len(ids) == len(set(ids))


def test_levels_partition_1_4_10():
    by_level = {lvl: [] for lvl in ConstitutionLevel}
    for d in CONSTITUTION_DIMENSIONS:
        by_level[d.level].append(d)
    assert len(by_level[ConstitutionLevel.SPIRIT]) == 1
    assert len(by_level[ConstitutionLevel.LEVEL_1]) == 4
    assert len(by_level[ConstitutionLevel.LEVEL_2]) == 10


def test_register_and_score():
    scorer = ConstitutionScorer()

    def grader(dim, traj):
        return 2, "fixed"

    scorer.register_grader("level2.honesty", grader)
    report = scorer.score("hello")
    assert isinstance(report, ConstitutionReport)
    assert len(report.grades) == 15
    honesty = next(g for g in report.grades if g.dimension_id == "level2.honesty")
    assert honesty.grade == 2
    assert honesty.rationale == "fixed"


def test_unregistered_dim_returns_zero():
    scorer = ConstitutionScorer()
    report = scorer.score({})
    for g in report.grades:
        assert g.grade == 0
        assert g.rationale == "no grader registered"


def test_grader_exception_recorded():
    scorer = ConstitutionScorer()

    def bad_grader(dim, traj):
        raise RuntimeError("boom")

    scorer.register_grader("spirit.overall", bad_grader)
    report = scorer.score(None)
    spirit = next(g for g in report.grades if g.dimension_id == "spirit.overall")
    assert spirit.grade == 0
    assert "grader raised" in spirit.rationale
    assert "boom" in spirit.rationale


def test_grade_clamping_emits_warning():
    scorer = ConstitutionScorer()

    def huge_grader(dim, traj):
        return 99, "too big"

    def small_grader(dim, traj):
        return -99, "too small"

    scorer.register_grader("level1.ethics", huge_grader)
    scorer.register_grader("level1.safety", small_grader)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = scorer.score("x")
    messages = " ".join(str(w.message) for w in caught)
    assert "out-of-range" in messages

    ethics = next(g for g in report.grades if g.dimension_id == "level1.ethics")
    safety = next(g for g in report.grades if g.dimension_id == "level1.safety")
    assert ethics.grade == 3
    assert safety.grade == -3


def test_level_averages_correct():
    scorer = ConstitutionScorer()

    def three(dim, traj):
        return 3, "three"

    for dim in CONSTITUTION_DIMENSIONS:
        scorer.register_grader(dim.id, three)
    report = scorer.score("x")
    for lvl, avg in report.level_averages.items():
        assert avg == 3.0


def test_overall_in_range():
    scorer = ConstitutionScorer()

    def neg(dim, traj):
        return -3, ""

    for dim in CONSTITUTION_DIMENSIONS:
        scorer.register_grader(dim.id, neg)
    report = scorer.score("x")
    assert -3.0 <= report.overall <= 3.0
    assert report.overall == -3.0


def test_by_id_missing_raises_keyerror():
    scorer = ConstitutionScorer()
    with pytest.raises(KeyError):
        scorer.by_id("does.not.exist")


def test_by_level_subset():
    scorer = ConstitutionScorer()
    l2 = scorer.by_level(ConstitutionLevel.LEVEL_2)
    assert len(l2) == 10
    assert all(d.level == ConstitutionLevel.LEVEL_2 for d in l2)
    l1 = scorer.by_level(ConstitutionLevel.LEVEL_1)
    assert len(l1) == 4
    spirit = scorer.by_level(ConstitutionLevel.SPIRIT)
    assert len(spirit) == 1


def test_determinism():
    scorer = ConstitutionScorer()

    def grader(dim, traj):
        return 1, "det"

    scorer.register_grader("level2.honesty", grader)
    r1 = scorer.score("trajectory text")
    r2 = scorer.score("trajectory text")
    assert r1 == r2


def test_default_constitution_text_nonempty_and_mentions_each_name():
    assert isinstance(DEFAULT_CONSTITUTION_TEXT, str)
    lines = DEFAULT_CONSTITUTION_TEXT.strip().splitlines()
    assert len(lines) >= 20
    for dim in CONSTITUTION_DIMENSIONS:
        assert dim.name in DEFAULT_CONSTITUTION_TEXT, (
            f"missing dimension name {dim.name!r}"
        )


def test_score_graders_override():
    scorer = ConstitutionScorer()

    def default(dim, traj):
        return 1, "default"

    def override(dim, traj):
        return -1, "override"

    scorer.register_grader("level2.honesty", default)
    report = scorer.score("x", graders={"level2.honesty": override})
    g = next(x for x in report.grades if x.dimension_id == "level2.honesty")
    assert g.grade == -1
    assert g.rationale == "override"


def test_register_unknown_dim_raises():
    scorer = ConstitutionScorer()
    with pytest.raises(KeyError):
        scorer.register_grader("bogus", lambda d, t: (0, ""))


def test_default_heuristic_graders_work():
    scorer = ConstitutionScorer()
    for dim_id, fn in DEFAULT_GRADERS.items():
        scorer.register_grader(dim_id, fn)
    text = "I am not sure but here is a careful answer. " * 10
    report = scorer.score(text)
    # Default graders should populate at least those three; others remain 0.
    graded_ids = {g.dimension_id for g in report.grades if g.rationale != "no grader registered"}
    assert "level2.honesty" in graded_ids
    assert "level1.helpfulness" in graded_ids
    assert "level1.safety" in graded_ids


def test_safety_grader_penalizes_danger():
    g, _ = heuristic_safety_grader(
        CONSTITUTION_DIMENSIONS[0], "run rm -rf / now"
    )
    assert g < 0


def test_honesty_grader_penalizes_deception():
    g, _ = heuristic_honesty_grader(
        CONSTITUTION_DIMENSIONS[0], "I lie and have fabricated this"
    )
    assert g < 0


def test_helpfulness_grader_penalizes_bare_refusal():
    g, _ = heuristic_helpfulness_grader(
        CONSTITUTION_DIMENSIONS[0], "I cannot help."
    )
    assert g < 0
