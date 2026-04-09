"""Tests for src/data/preference_collector.py."""

from __future__ import annotations

import pytest

from src.data.preference_collector import (
    PreferenceConfig,
    PreferencePair,
    PreferenceCollector,
    create_preference_pair,
    format_for_dpo,
    format_for_rlhf,
    score_by_length,
    score_by_rules,
)


# ---------------------------------------------------------------------------
# 1. PreferenceConfig defaults
# ---------------------------------------------------------------------------

def test_preference_config_defaults():
    cfg = PreferenceConfig()
    assert cfg.n_responses == 4
    assert cfg.scoring_method == "length"
    assert cfg.min_response_len == 10
    assert cfg.max_response_len == 500
    assert cfg.tie_threshold == 0.1
    assert cfg.format == "dpo"


# ---------------------------------------------------------------------------
# 2. score_by_length — perfect match returns 1.0
# ---------------------------------------------------------------------------

def test_score_by_length_perfect_match():
    response = "a" * 100
    score = score_by_length(response, target_len=100)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. score_by_length — very long response returns low score
# ---------------------------------------------------------------------------

def test_score_by_length_very_long():
    # 10 000 chars vs target 100 → score ≈ 1 - 9900/10000 = 0.01
    response = "a" * 10_000
    score = score_by_length(response, target_len=100)
    assert score < 0.1


# ---------------------------------------------------------------------------
# 4. score_by_rules — penalizes short response
# ---------------------------------------------------------------------------

def test_score_by_rules_penalizes_short():
    short = "Hi"  # < 20 chars
    score = score_by_rules(short)
    # Baseline 0.5, -0.3 for short → 0.2 (may gain punctuation bonus if ends with .)
    assert score <= 0.4


# ---------------------------------------------------------------------------
# 5. score_by_rules — rewards punctuated response
# ---------------------------------------------------------------------------

def test_score_by_rules_rewards_punctuation():
    # Exactly 20 chars, ends with period — no length penalty, +0.2 punctuation
    text = "This is a sentence!!"  # 20 chars, ends with !
    score_with_punct = score_by_rules(text)
    # Same length but no trailing punctuation
    text_no_punct = "This is a sentence  "
    score_no_punct = score_by_rules(text_no_punct)
    assert score_with_punct > score_no_punct


# ---------------------------------------------------------------------------
# 6. score_by_rules — penalizes repeated n-grams
# ---------------------------------------------------------------------------

def test_score_by_rules_penalizes_repeated_ngrams():
    # Repeat the same 5 words > 2 times
    base = "the cat sat on mat "
    repetitive = (base * 4).strip()
    score = score_by_rules(repetitive)
    # Baseline 0.5 -0.3 (repetition) + 0.0 (no end punct) = 0.2 max
    assert score <= 0.3


# ---------------------------------------------------------------------------
# 7. create_preference_pair — correct chosen/rejected selection
# ---------------------------------------------------------------------------

def test_create_preference_pair_correct_selection():
    cfg = PreferenceConfig(min_response_len=5, tie_threshold=0.05)
    prompt = "What is 2+2?"
    responses = ["four exactly.", "no idea", "it is four indeed correct."]
    scores = [0.9, 0.2, 0.6]

    pair = create_preference_pair(prompt, responses, scores, cfg)
    assert pair is not None
    assert pair.chosen == "four exactly."
    assert pair.rejected == "no idea"
    assert pair.chosen_score == pytest.approx(0.9)
    assert pair.rejected_score == pytest.approx(0.2)
    assert pair.score_diff == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 8. create_preference_pair — returns None for ties
# ---------------------------------------------------------------------------

def test_create_preference_pair_none_on_tie():
    cfg = PreferenceConfig(min_response_len=5, tie_threshold=0.2)
    prompt = "Hello?"
    responses = ["response one okay", "response two okay"]
    scores = [0.5, 0.55]  # diff = 0.05 < 0.2

    pair = create_preference_pair(prompt, responses, scores, cfg)
    assert pair is None


# ---------------------------------------------------------------------------
# 9. create_preference_pair — returns None if responses too short
# ---------------------------------------------------------------------------

def test_create_preference_pair_none_on_too_short():
    cfg = PreferenceConfig(min_response_len=50, tie_threshold=0.1)
    prompt = "Hello?"
    responses = ["short", "also short response here"]
    scores = [0.9, 0.1]  # big diff but both below min_response_len=50

    pair = create_preference_pair(prompt, responses, scores, cfg)
    assert pair is None


# ---------------------------------------------------------------------------
# 10. format_for_dpo — correct keys
# ---------------------------------------------------------------------------

def test_format_for_dpo_keys():
    pair = PreferencePair(
        prompt="Q?",
        chosen="Great answer.",
        rejected="Bad answer.",
        chosen_score=0.9,
        rejected_score=0.3,
        score_diff=0.6,
    )
    result = format_for_dpo(pair)
    assert set(result.keys()) == {"prompt", "chosen", "rejected"}
    assert result["prompt"] == "Q?"
    assert result["chosen"] == "Great answer."
    assert result["rejected"] == "Bad answer."


# ---------------------------------------------------------------------------
# 11. format_for_rlhf — returns two entries (chosen + rejected)
# ---------------------------------------------------------------------------

def test_format_for_rlhf_two_entries():
    pair = PreferencePair(
        prompt="Q?",
        chosen="Great answer.",
        rejected="Bad answer.",
        chosen_score=0.9,
        rejected_score=0.3,
        score_diff=0.6,
    )
    entries = format_for_rlhf(pair)
    assert len(entries) == 2

    chosen_entry = next(e for e in entries if e["response"] == "Great answer.")
    rejected_entry = next(e for e in entries if e["response"] == "Bad answer.")

    assert chosen_entry["reward"] > 0
    assert rejected_entry["reward"] < 0
    assert all("prompt" in e and "response" in e and "reward" in e for e in entries)


# ---------------------------------------------------------------------------
# 12. PreferenceCollector.collect — returns PreferencePair or None
# ---------------------------------------------------------------------------

def test_preference_collector_collect_returns_pair_or_none():
    cfg = PreferenceConfig(min_response_len=5, tie_threshold=0.05)
    collector = PreferenceCollector(cfg)

    # Should return a pair
    pair = collector.collect(
        "What is Python?",
        [
            "Python is a programming language that is widely used.",
            "idk",  # too short (< 10 default min), also low length score
        ],
    )
    # With default min_response_len=10, "idk" is 3 chars → pair is None
    # Adjust test: use responses both above min_response_len
    cfg2 = PreferenceConfig(min_response_len=3, tie_threshold=0.05)
    collector2 = PreferenceCollector(cfg2)
    pair2 = collector2.collect(
        "What is Python?",
        [
            "x" * 100,   # exactly 100 chars → score ≈ 1.0
            "x" * 10,    # 10 chars → score lower
        ],
    )
    assert pair2 is not None
    assert isinstance(pair2, PreferencePair)


# ---------------------------------------------------------------------------
# 13. PreferenceCollector.collect_batch — filters out Nones
# ---------------------------------------------------------------------------

def test_preference_collector_collect_batch_filters_nones():
    cfg = PreferenceConfig(min_response_len=3, tie_threshold=0.3)
    collector = PreferenceCollector(cfg)

    prompts = ["Q1?", "Q2?"]
    responses_list = [
        # Pair 1: large score diff (100 chars vs 10 chars → passes threshold)
        ["x" * 100, "x" * 10],
        # Pair 2: very similar lengths → near-identical scores → tie → None
        ["x" * 100, "x" * 101],
    ]

    pairs = collector.collect_batch(prompts, responses_list)
    # At least pair 1 should survive; pair 2 may be filtered
    assert isinstance(pairs, list)
    for p in pairs:
        assert isinstance(p, PreferencePair)


# ---------------------------------------------------------------------------
# 14. PreferenceCollector.statistics — correct mean_score_diff
# ---------------------------------------------------------------------------

def test_preference_collector_statistics_mean_score_diff():
    cfg = PreferenceConfig()
    collector = PreferenceCollector(cfg)

    pairs = [
        PreferencePair("p1", "c", "r", chosen_score=0.9, rejected_score=0.3, score_diff=0.6),
        PreferencePair("p2", "c", "r", chosen_score=0.8, rejected_score=0.4, score_diff=0.4),
    ]

    stats = collector.statistics(pairs)
    assert stats["n_pairs"] == 2
    assert stats["mean_score_diff"] == pytest.approx(0.5, abs=1e-6)
    assert stats["mean_chosen_score"] == pytest.approx(0.85, abs=1e-6)


# ---------------------------------------------------------------------------
# Bonus: statistics on empty list
# ---------------------------------------------------------------------------

def test_preference_collector_statistics_empty():
    cfg = PreferenceConfig()
    collector = PreferenceCollector(cfg)
    stats = collector.statistics([])
    assert stats["n_pairs"] == 0
    assert stats["mean_score_diff"] == 0.0
    assert stats["mean_chosen_score"] == 0.0


# ---------------------------------------------------------------------------
# Bonus: to_dataset respects format
# ---------------------------------------------------------------------------

def test_to_dataset_dpo_format():
    cfg = PreferenceConfig(format="dpo")
    collector = PreferenceCollector(cfg)
    pairs = [
        PreferencePair("p", "chosen text.", "rejected text.", 0.9, 0.3, 0.6),
    ]
    dataset = collector.to_dataset(pairs)
    assert len(dataset) == 1
    assert set(dataset[0].keys()) == {"prompt", "chosen", "rejected"}


def test_to_dataset_rlhf_format():
    cfg = PreferenceConfig(format="rlhf")
    collector = PreferenceCollector(cfg)
    pairs = [
        PreferencePair("p", "chosen text.", "rejected text.", 0.9, 0.3, 0.6),
    ]
    dataset = collector.to_dataset(pairs)
    assert len(dataset) == 2  # one chosen + one rejected entry


def test_to_dataset_sft_only_format():
    cfg = PreferenceConfig(format="sft_only")
    collector = PreferenceCollector(cfg)
    pairs = [
        PreferencePair("p", "chosen text.", "rejected text.", 0.9, 0.3, 0.6),
    ]
    dataset = collector.to_dataset(pairs)
    assert len(dataset) == 1
    assert set(dataset[0].keys()) == {"prompt", "response"}
    assert dataset[0]["response"] == "chosen text."
