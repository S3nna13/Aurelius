"""Tests for RedTeamGenerator."""
from __future__ import annotations

import json

import pytest

from src.safety.red_team_generator import (
    AttackCategory,
    RedTeamCase,
    RedTeamGenerator,
    RedTeamReport,
)


@pytest.fixture()
def gen() -> RedTeamGenerator:
    return RedTeamGenerator(seed=42)


def test_generate_returns_correct_count(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.JAILBREAK, n=5)
    assert len(cases) == 5


def test_generate_correct_category(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.DIRECT_HARM, n=3)
    for case in cases:
        assert case.category == AttackCategory.DIRECT_HARM


def test_generate_case_ids_are_unique(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.JAILBREAK, n=10)
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_generate_severity_in_range(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.MISINFORMATION, n=20)
    for case in cases:
        assert 0.3 <= case.severity <= 1.0


def test_generate_prompt_is_nonempty(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.BIAS_ELICITATION, n=3)
    for case in cases:
        assert len(case.prompt) > 0


def test_generate_tags_populated(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.MANIPULATION, n=2)
    for case in cases:
        assert isinstance(case.tags, list)
        assert len(case.tags) > 0


def test_generate_suite_covers_all_categories(gen: RedTeamGenerator) -> None:
    report = gen.generate_suite(n_per_category=2)
    expected_categories = {cat.value for cat in AttackCategory}
    assert set(report.cases_by_category.keys()) == expected_categories


def test_generate_suite_total_count(gen: RedTeamGenerator) -> None:
    n = 3
    report = gen.generate_suite(n_per_category=n)
    assert report.n_cases == n * len(AttackCategory)


def test_generate_suite_mean_severity_in_range(gen: RedTeamGenerator) -> None:
    report = gen.generate_suite(n_per_category=5)
    assert 0.3 <= report.mean_severity <= 1.0


def test_generate_suite_returns_report_type(gen: RedTeamGenerator) -> None:
    report = gen.generate_suite()
    assert isinstance(report, RedTeamReport)


def test_filter_by_severity_removes_low(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.JAILBREAK, n=20)
    filtered = gen.filter_by_severity(cases, min_severity=0.7)
    for case in filtered:
        assert case.severity >= 0.7


def test_filter_by_severity_all_pass(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.DIRECT_HARM, n=5)
    filtered = gen.filter_by_severity(cases, min_severity=0.0)
    assert len(filtered) == len(cases)


def test_filter_by_severity_none_pass(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.PRIVACY_VIOLATION, n=5)
    filtered = gen.filter_by_severity(cases, min_severity=1.1)
    assert filtered == []


def test_export_jsonl_valid_json_lines(gen: RedTeamGenerator) -> None:
    cases = gen.generate(AttackCategory.MANIPULATION, n=3)
    jsonl = gen.export_jsonl(cases)
    lines = jsonl.strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert "case_id" in record
        assert "category" in record
        assert "prompt" in record
        assert "severity" in record


def test_export_jsonl_empty_list(gen: RedTeamGenerator) -> None:
    result = gen.export_jsonl([])
    assert result == ""


def test_seeded_reproducibility() -> None:
    g1 = RedTeamGenerator(seed=99)
    g2 = RedTeamGenerator(seed=99)
    cases1 = g1.generate(AttackCategory.JAILBREAK, n=5)
    cases2 = g2.generate(AttackCategory.JAILBREAK, n=5)
    prompts1 = [c.prompt for c in cases1]
    prompts2 = [c.prompt for c in cases2]
    assert prompts1 == prompts2
