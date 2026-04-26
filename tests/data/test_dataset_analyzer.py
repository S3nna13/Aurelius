"""Tests for src/data/dataset_analyzer.py"""

from src.data.dataset_analyzer import (
    DATASET_ANALYZER_REGISTRY,
    DatasetAnalyzer,
    DatasetStats,
)


def make_examples():
    return [
        {"prompt": "What is 2+2?", "response": "It is four.", "domain": "math"},
        {"prompt": "What is Python?", "response": "A programming language.", "domain": "coding"},
        {"prompt": "What is logic?", "response": "Reasoning with rules.", "domain": "logic"},
        {"prompt": "What is AI?", "response": "Artificial intelligence.", "domain": "general"},
    ]


def test_analyze_returns_dataset_stats():
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze(make_examples())
    assert isinstance(stats, DatasetStats)


def test_analyze_n_examples():
    analyzer = DatasetAnalyzer()
    examples = make_examples()
    stats = analyzer.analyze(examples)
    assert stats.n_examples == len(examples)


def test_analyze_avg_prompt_len():
    analyzer = DatasetAnalyzer()
    examples = [{"prompt": "hello", "response": "world", "domain": ""}]
    stats = analyzer.analyze(examples)
    assert stats.avg_prompt_len == 5.0


def test_analyze_max_min_prompt_len():
    analyzer = DatasetAnalyzer()
    examples = [
        {"prompt": "ab", "response": "", "domain": ""},
        {"prompt": "abcde", "response": "", "domain": ""},
    ]
    stats = analyzer.analyze(examples)
    assert stats.max_prompt_len == 5
    assert stats.min_prompt_len == 2


def test_analyze_vocab_size():
    analyzer = DatasetAnalyzer()
    examples = [
        {"prompt": "hello world", "response": "foo bar", "domain": ""},
        {"prompt": "hello again", "response": "baz", "domain": ""},
    ]
    stats = analyzer.analyze(examples)
    assert stats.vocab_size == len({"hello", "world", "foo", "bar", "again", "baz"})


def test_analyze_domain_distribution():
    analyzer = DatasetAnalyzer()
    examples = make_examples()
    stats = analyzer.analyze(examples)
    assert stats.domain_distribution.get("math") == 1
    assert stats.domain_distribution.get("coding") == 1


def test_analyze_empty():
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze([])
    assert stats.n_examples == 0
    assert stats.vocab_size == 0


def test_length_histogram_n_bins():
    analyzer = DatasetAnalyzer()
    examples = [{"prompt": "a" * i, "response": ""} for i in range(1, 21)]
    hist = analyzer.length_histogram(examples, key="prompt", n_bins=5)
    assert len(hist) == 5


def test_length_histogram_counts_sum_to_total():
    analyzer = DatasetAnalyzer()
    examples = [{"prompt": "a" * i, "response": ""} for i in range(1, 11)]
    hist = analyzer.length_histogram(examples, key="prompt", n_bins=5)
    assert sum(hist.values()) == len(examples)


def test_length_histogram_keys_are_strings():
    analyzer = DatasetAnalyzer()
    examples = [{"prompt": "hello", "response": "world"}]
    hist = analyzer.length_histogram(examples, key="prompt", n_bins=10)
    for k in hist:
        assert isinstance(k, str)


def test_detect_duplicates_finds_indices():
    analyzer = DatasetAnalyzer()
    examples = [
        {"prompt": "same"},
        {"prompt": "different"},
        {"prompt": "same"},
        {"prompt": "another"},
        {"prompt": "same"},
    ]
    dupes = analyzer.detect_duplicates(examples)
    assert 2 in dupes
    assert 4 in dupes
    assert 0 not in dupes


def test_detect_duplicates_empty_when_no_dupes():
    analyzer = DatasetAnalyzer()
    examples = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    dupes = analyzer.detect_duplicates(examples)
    assert dupes == []


def test_quality_score_clips_to_one():
    analyzer = DatasetAnalyzer()
    ex = {"prompt": "hi", "response": "x" * 100}
    score = analyzer.quality_score(ex)
    assert score <= 1.0


def test_quality_score_minimum_zero():
    analyzer = DatasetAnalyzer()
    ex = {"prompt": "x" * 100, "response": ""}
    score = analyzer.quality_score(ex)
    assert score >= 0.0


def test_quality_score_good_threshold():
    analyzer = DatasetAnalyzer()
    ex = {"prompt": "short", "response": "a much longer response here"}
    score = analyzer.quality_score(ex)
    assert score > 0.5


def test_registry_has_default():
    assert "default" in DATASET_ANALYZER_REGISTRY
    assert DATASET_ANALYZER_REGISTRY["default"] is DatasetAnalyzer
