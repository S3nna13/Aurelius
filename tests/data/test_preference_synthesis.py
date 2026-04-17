"""Tests for src/data/preference_synthesis.py — mock data only, no network."""

from __future__ import annotations

import pytest

from src.data.preference_synthesis import (
    ArgillaSample,
    NectarSample,
    OpenHermesSample,
    PreferenceSynthesizer,
    mock_argilla_data,
    mock_nectar_data,
    mock_openhermes_data,
    parse_argilla_sample,
    parse_nectar_sample,
    parse_openhermes_sample,
)


# ---------------------------------------------------------------------------
# 1. mock_nectar_data has id, prompt, answers fields
# ---------------------------------------------------------------------------

def test_mock_nectar_data_fields():
    data = mock_nectar_data(n=4)
    assert len(data) == 4
    for item in data:
        assert "id" in item
        assert "prompt" in item
        assert "answers" in item
        assert isinstance(item["answers"], list)


# ---------------------------------------------------------------------------
# 2. NectarSample.best_answer returns rank=1 answer
# ---------------------------------------------------------------------------

def test_nectar_best_answer_rank_one():
    raw = mock_nectar_data(n=1)[0]
    sample = parse_nectar_sample(raw)
    best = sample.best_answer()
    rank_one_text = next(a["answer"] for a in raw["answers"] if a["rank"] == 1)
    assert best == rank_one_text


# ---------------------------------------------------------------------------
# 3. NectarSample.worst_answer returns last-ranked answer
# ---------------------------------------------------------------------------

def test_nectar_worst_answer():
    raw = mock_nectar_data(n=1)[0]
    sample = parse_nectar_sample(raw)
    worst = sample.worst_answer()
    max_rank = max(a["rank"] for a in raw["answers"])
    worst_text = next(a["answer"] for a in raw["answers"] if a["rank"] == max_rank)
    assert worst == worst_text


# ---------------------------------------------------------------------------
# 4. NectarSample.to_preference_pair has prompt/chosen/rejected
# ---------------------------------------------------------------------------

def test_nectar_to_preference_pair_keys():
    sample = parse_nectar_sample(mock_nectar_data(n=1)[0])
    pair = sample.to_preference_pair()
    assert "prompt" in pair
    assert "chosen" in pair
    assert "rejected" in pair
    assert pair["chosen"] != pair["rejected"]


# ---------------------------------------------------------------------------
# 5. mock_openhermes_data has conversations field
# ---------------------------------------------------------------------------

def test_mock_openhermes_data_conversations():
    data = mock_openhermes_data(n=4)
    assert len(data) == 4
    for item in data:
        assert "conversations" in item
        assert isinstance(item["conversations"], list)
        assert len(item["conversations"]) >= 2


# ---------------------------------------------------------------------------
# 6. OpenHermesSample.to_instruction extracts human/gpt turns
# ---------------------------------------------------------------------------

def test_openhermes_to_instruction():
    raw = mock_openhermes_data(n=1)[0]
    sample = parse_openhermes_sample(raw)
    inst = sample.to_instruction()
    assert "instruction" in inst
    assert "output" in inst
    assert inst["instruction"] != ""
    assert inst["output"] != ""
    # instruction should come from the human turn
    human_value = next(c["value"] for c in raw["conversations"] if c["from"] == "human")
    assert inst["instruction"] == human_value


# ---------------------------------------------------------------------------
# 7. mock_argilla_data has chosen, rejected, ratings
# ---------------------------------------------------------------------------

def test_mock_argilla_data_fields():
    data = mock_argilla_data(n=4)
    assert len(data) == 4
    for item in data:
        assert "chosen" in item
        assert "rejected" in item
        assert "raw_chosen" in item
        assert "raw_rejected" in item
        assert "rating" in item["raw_chosen"]
        assert "rating" in item["raw_rejected"]


# ---------------------------------------------------------------------------
# 8. parse_argilla_sample chosen_rating is float
# ---------------------------------------------------------------------------

def test_argilla_chosen_rating_is_float():
    raw = mock_argilla_data(n=1)[0]
    sample = parse_argilla_sample(raw)
    assert isinstance(sample.chosen_rating, float)
    assert isinstance(sample.rejected_rating, float)


# ---------------------------------------------------------------------------
# 9. ArgillaSample chosen_rating > rejected_rating in mock
# ---------------------------------------------------------------------------

def test_argilla_chosen_rating_greater_than_rejected():
    for raw in mock_argilla_data(n=4):
        sample = parse_argilla_sample(raw)
        assert sample.chosen_rating > sample.rejected_rating


# ---------------------------------------------------------------------------
# 10. PreferenceSynthesizer.score_response returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_score_response_range():
    synth = PreferenceSynthesizer(seed=0)
    test_cases = [
        "",
        "Hello.",
        "A " * 200,  # ~200 words
        "# Header\n- bullet\n```code```\n" + "Word " * 150,
    ]
    for text in test_cases:
        score = synth.score_response(text)
        assert isinstance(score, float), f"Expected float for: {text[:30]!r}"
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for: {text[:30]!r}"


# ---------------------------------------------------------------------------
# 11. Longer/structured responses score higher than empty strings
# ---------------------------------------------------------------------------

def test_score_response_structured_beats_empty():
    synth = PreferenceSynthesizer(seed=0)
    empty_score = synth.score_response("")
    structured = (
        "# Introduction\n"
        "This is a detailed response that covers multiple aspects.\n"
        "- Point one is important\n"
        "- Point two builds on that\n"
        "```python\nprint('hello')\n```\n"
        + "Additional content " * 20
    )
    structured_score = synth.score_response(structured)
    assert structured_score > empty_score


def test_score_response_longer_beats_single_word():
    synth = PreferenceSynthesizer(seed=0)
    single = synth.score_response("Yes.")
    longer = synth.score_response(
        "This is a comprehensive answer that explains the topic in detail. "
        "It includes multiple sentences and covers various aspects. " * 10
    )
    assert longer > single


# ---------------------------------------------------------------------------
# 12. create_pair_from_instructions returns list of dicts with correct keys
# ---------------------------------------------------------------------------

def test_create_pair_from_instructions_keys():
    synth = PreferenceSynthesizer(seed=42)
    samples = [
        {
            "instruction": f"Explain topic {i}.",
            "input": "",
            "output": ("This is a thorough explanation. " * (i + 1)).strip(),
        }
        for i in range(8)
    ]
    pairs = synth.create_pair_from_instructions(samples, n_pairs=4)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for pair in pairs:
        assert "prompt" in pair, f"Missing 'prompt' in {pair}"
        assert "chosen" in pair, f"Missing 'chosen' in {pair}"
        assert "rejected" in pair, f"Missing 'rejected' in {pair}"


def test_create_pair_from_instructions_empty_input():
    synth = PreferenceSynthesizer(seed=42)
    result = synth.create_pair_from_instructions([], n_pairs=4)
    assert result == []


# ---------------------------------------------------------------------------
# 13. augment_with_negatives truncate → rejected shorter than chosen
# ---------------------------------------------------------------------------

def test_augment_truncate_shorter():
    synth = PreferenceSynthesizer(seed=0)
    pairs = [
        {
            "prompt": "Tell me about X.",
            "chosen": "This is a detailed and lengthy response about X. " * 5,
            "rejected": "X is a thing.",
        }
    ]
    augmented = synth.augment_with_negatives(pairs, strategies=["truncate"])
    assert len(augmented) == 1
    assert len(augmented[0]["rejected"]) < len(augmented[0]["chosen"])


# ---------------------------------------------------------------------------
# 14. augment_with_negatives returns same length list as input
# ---------------------------------------------------------------------------

def test_augment_with_negatives_same_length():
    synth = PreferenceSynthesizer(seed=7)
    pairs = [
        {
            "prompt": f"Question {i}?",
            "chosen": f"Detailed answer to question {i}. " * 3,
            "rejected": f"Short answer {i}.",
        }
        for i in range(6)
    ]
    augmented = synth.augment_with_negatives(pairs)
    assert len(augmented) == len(pairs)


def test_augment_all_strategies_return_string():
    synth = PreferenceSynthesizer(seed=1)
    chosen_text = "The first sentence is here. Then more content follows. And even more."
    for strategy in ("truncate", "repeat", "shuffle_words"):
        pairs = [{"prompt": "Q?", "chosen": chosen_text, "rejected": "old"}]
        result = synth.augment_with_negatives(pairs, strategies=[strategy])
        assert isinstance(result[0]["rejected"], str)
        assert result[0]["chosen"] == chosen_text
