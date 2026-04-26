"""Tests for src/data/data_quality_checker.py"""

import pytest

from src.data.data_quality_checker import (
    DataQualityChecker,
    QualityConfig,
    QualityIssue,
    QualityReport,
)

# ---------------------------------------------------------------------------
# QualityConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = QualityConfig()
    assert cfg.min_chars == 10
    assert cfg.max_chars == 100_000
    assert cfg.repetition_threshold == 0.4
    assert cfg.entropy_threshold == 2.5
    assert cfg.toxic_keywords == []


def test_config_custom():
    cfg = QualityConfig(min_chars=5, max_chars=500, toxic_keywords=["spam"])
    assert cfg.min_chars == 5
    assert cfg.max_chars == 500
    assert cfg.toxic_keywords == ["spam"]


# ---------------------------------------------------------------------------
# QualityReport structure
# ---------------------------------------------------------------------------


def test_quality_report_passed_when_no_issues():
    report = QualityReport(text="hello", issues=[], passed=True, stats={})
    assert report.passed is True
    assert report.issues == []


def test_quality_report_failed_when_issues_present():
    report = QualityReport(
        text="hi",
        issues=[QualityIssue.TOO_SHORT],
        passed=False,
        stats={},
    )
    assert report.passed is False
    assert QualityIssue.TOO_SHORT in report.issues


# ---------------------------------------------------------------------------
# _char_entropy
# ---------------------------------------------------------------------------


def test_entropy_empty_string():
    checker = DataQualityChecker()
    assert checker._char_entropy("") == 0.0


def test_entropy_uniform_string():
    checker = DataQualityChecker()
    entropy = checker._char_entropy("aaaa")
    assert entropy == pytest.approx(0.0)


def test_entropy_two_chars_equal_split():
    checker = DataQualityChecker()
    entropy = checker._char_entropy("ab" * 50)
    assert entropy == pytest.approx(1.0, abs=0.01)


def test_entropy_high_for_diverse_text():
    checker = DataQualityChecker()
    text = "The quick brown fox jumps over the lazy dog"
    assert checker._char_entropy(text) > 3.0


# ---------------------------------------------------------------------------
# _repetition_ratio
# ---------------------------------------------------------------------------


def test_repetition_ratio_zero_for_unique_ngrams():
    checker = DataQualityChecker()
    text = "the quick brown fox jumps over lazy dogs here now"
    ratio = checker._repetition_ratio(text)
    assert ratio == pytest.approx(0.0)


def test_repetition_ratio_high_for_repeated_phrase():
    checker = DataQualityChecker()
    text = " ".join(["hello world foo bar"] * 10)
    ratio = checker._repetition_ratio(text)
    assert ratio > 0.7


def test_repetition_ratio_short_text_returns_zero():
    checker = DataQualityChecker()
    ratio = checker._repetition_ratio("hi")
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# check: individual issues
# ---------------------------------------------------------------------------


def test_check_too_short():
    checker = DataQualityChecker(QualityConfig(min_chars=20))
    report = checker.check("short")
    assert QualityIssue.TOO_SHORT in report.issues
    assert report.passed is False


def test_check_too_long():
    checker = DataQualityChecker(QualityConfig(max_chars=5))
    report = checker.check("this is way too long for the limit")
    assert QualityIssue.TOO_LONG in report.issues
    assert report.passed is False


def test_check_high_repetition():
    checker = DataQualityChecker(
        QualityConfig(
            min_chars=1,
            entropy_threshold=0.0,
            repetition_threshold=0.3,
        )
    )
    text = " ".join(["the same four words"] * 20)
    report = checker.check(text)
    assert QualityIssue.HIGH_REPETITION in report.issues


def test_check_low_entropy():
    checker = DataQualityChecker(QualityConfig(min_chars=1, entropy_threshold=3.0))
    report = checker.check("aaaaaaaaaa")
    assert QualityIssue.LOW_ENTROPY in report.issues


def test_check_toxic_keyword_case_insensitive():
    checker = DataQualityChecker(
        QualityConfig(
            min_chars=1,
            entropy_threshold=0.0,
            toxic_keywords=["badword"],
        )
    )
    report = checker.check("This contains BADWORD inside it for testing purposes.")
    assert QualityIssue.TOXIC_KEYWORDS in report.issues


def test_check_no_issues_for_clean_text():
    checker = DataQualityChecker()
    text = (
        "Machine learning models require large datasets to generalize well. "
        "Transformer architectures have revolutionized natural language processing. "
        "Attention mechanisms allow models to focus on relevant tokens dynamically."
    )
    report = checker.check(text)
    assert report.passed is True
    assert report.issues == []


def test_check_stats_keys_present():
    checker = DataQualityChecker()
    text = "The quick brown fox jumps over the lazy dog."
    report = checker.check(text)
    assert "char_count" in report.stats
    assert "word_count" in report.stats
    assert "entropy" in report.stats
    assert "repetition_ratio" in report.stats


def test_check_stats_values_correct():
    checker = DataQualityChecker()
    text = "hello world"
    report = checker.check(text)
    assert report.stats["char_count"] == 11
    assert report.stats["word_count"] == 2


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_returns_only_passing_texts():
    checker = DataQualityChecker(QualityConfig(min_chars=20, entropy_threshold=0.0))
    texts = [
        "short",
        "The quick brown fox jumps over the lazy dog and more words here.",
    ]
    result = checker.filter(texts)
    assert len(result) == 1
    assert result[0].startswith("The quick")


def test_filter_empty_list():
    checker = DataQualityChecker()
    assert checker.filter([]) == []


def test_filter_all_pass():
    checker = DataQualityChecker(QualityConfig(min_chars=1, entropy_threshold=0.0))
    texts = ["hello world", "foo bar baz"]
    result = checker.filter(texts)
    assert result == texts


# ---------------------------------------------------------------------------
# batch_check
# ---------------------------------------------------------------------------


def test_batch_check_returns_list_of_reports():
    checker = DataQualityChecker()
    texts = ["hello world foo bar baz test", "hi"]
    reports = checker.batch_check(texts)
    assert len(reports) == 2
    assert all(isinstance(r, QualityReport) for r in reports)


def test_batch_check_empty():
    checker = DataQualityChecker()
    assert checker.batch_check([]) == []


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_counts():
    checker = DataQualityChecker(QualityConfig(min_chars=20, entropy_threshold=0.0))
    texts = [
        "short",
        "The quick brown fox jumps over the lazy dog, a longer sentence here.",
    ]
    reports = checker.batch_check(texts)
    s = checker.summary(reports)
    assert s["total"] == 2
    assert s["passed"] == 1
    assert s["failed"] == 1
    assert s["issue_counts"].get("too_short", 0) == 1


def test_summary_empty():
    checker = DataQualityChecker()
    s = checker.summary([])
    assert s["total"] == 0
    assert s["passed"] == 0
    assert s["failed"] == 0
    assert s["issue_counts"] == {}


def test_summary_all_pass():
    checker = DataQualityChecker(QualityConfig(min_chars=1, entropy_threshold=0.0))
    texts = ["hello world foo", "bar baz qux quux"]
    reports = checker.batch_check(texts)
    s = checker.summary(reports)
    assert s["passed"] == 2
    assert s["failed"] == 0
    assert s["issue_counts"] == {}
