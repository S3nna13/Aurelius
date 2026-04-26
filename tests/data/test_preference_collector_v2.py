"""Tests for src/data/preference_collector_v2.py — 16 tests."""

from __future__ import annotations

import random

import pytest

from src.data.preference_collector_v2 import (
    CollectionConfig,
    DatasetStats,
    PreferenceCollectorV2,
    PreferenceDatasetV2,
    PreferencePair,
    Response,
    compute_text_similarity,
    is_near_duplicate,
    mine_preference_pairs,
    score_with_reward_model,
)

# ---------------------------------------------------------------------------
# Mock helpers (as specified)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "Good response A.",
    "Decent response B.",
    "Bad response C.",
    "Another response D.",
]


def mock_generate(prompt: str) -> str:
    return random.choice(_MOCK_RESPONSES)


def mock_reward(prompt: str, response: str) -> float:
    # Higher score for longer responses
    return len(response) / 50.0


# ---------------------------------------------------------------------------
# 1. compute_text_similarity — identical strings → 1.0
# ---------------------------------------------------------------------------


def test_similarity_identical():
    assert compute_text_similarity("hello world", "hello world") == 1.0


# ---------------------------------------------------------------------------
# 2. compute_text_similarity — different strings → < 0.5
# ---------------------------------------------------------------------------


def test_similarity_different():
    result = compute_text_similarity("hello world", "goodbye moon")
    assert result < 0.5


# ---------------------------------------------------------------------------
# 3. is_near_duplicate — identical strings → True
# ---------------------------------------------------------------------------


def test_near_duplicate_identical():
    assert is_near_duplicate("this is a test", "this is a test") is True


# ---------------------------------------------------------------------------
# 4. is_near_duplicate — completely different strings → False
# ---------------------------------------------------------------------------


def test_near_duplicate_different():
    assert is_near_duplicate("alpha beta gamma", "delta epsilon zeta") is False


# ---------------------------------------------------------------------------
# 5. score_with_reward_model returns list of Response
# ---------------------------------------------------------------------------


def test_score_with_reward_model_returns_responses():
    prompt = "What is the capital of France?"
    responses = ["Paris is the capital.", "I don't know."]
    result = score_with_reward_model(prompt, responses, mock_reward)
    assert isinstance(result, list)
    assert all(isinstance(r, Response) for r in result)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 6. score_with_reward_model scores are not None
# ---------------------------------------------------------------------------


def test_score_with_reward_model_scores_not_none():
    prompt = "Test prompt."
    responses = ["Response one.", "Response two.", "Response three."]
    result = score_with_reward_model(prompt, responses, mock_reward)
    for r in result:
        assert r.score is not None


# ---------------------------------------------------------------------------
# 7. mine_preference_pairs: chosen.score > rejected.score for all pairs
# ---------------------------------------------------------------------------


def test_mine_pairs_chosen_higher_score():
    config = CollectionConfig(min_margin=0.0, max_pairs_per_prompt=10)
    responses = [
        Response(text="Short.", score=0.1),
        Response(text="A medium length response here.", score=0.6),
        Response(text="A longer response with more content in it.", score=0.9),
    ]
    pairs = mine_preference_pairs("test prompt", responses, config)
    assert len(pairs) > 0
    for pair in pairs:
        assert pair.chosen.score > pair.rejected.score


# ---------------------------------------------------------------------------
# 8. mine_preference_pairs: margin >= min_margin for all returned pairs
# ---------------------------------------------------------------------------


def test_mine_pairs_margin_filter():
    config = CollectionConfig(min_margin=0.2, max_pairs_per_prompt=10)
    responses = [
        Response(text="Short.", score=0.1),
        Response(text="Medium response text here.", score=0.5),
        Response(text="A much longer and detailed response that scores higher.", score=0.9),
    ]
    pairs = mine_preference_pairs("test prompt", responses, config)
    for pair in pairs:
        assert pair.margin >= config.min_margin


# ---------------------------------------------------------------------------
# 9. mine_preference_pairs: empty responses → empty list
# ---------------------------------------------------------------------------


def test_mine_pairs_empty_responses():
    config = CollectionConfig()
    result = mine_preference_pairs("test prompt", [], config)
    assert result == []


# ---------------------------------------------------------------------------
# 10. PreferenceDatasetV2 starts empty
# ---------------------------------------------------------------------------


def test_dataset_starts_empty():
    ds = PreferenceDatasetV2()
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# 11. PreferenceDatasetV2.add increments length
# ---------------------------------------------------------------------------


def test_dataset_add_increments_length():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    pair = PreferencePair(
        prompt="p",
        chosen=Response(text="chosen text response here", score=0.9),
        rejected=Response(text="rejected short text", score=0.3),
        margin=0.6,
    )
    assert ds.add(pair) is True
    assert len(ds) == 1


# ---------------------------------------------------------------------------
# 12. PreferenceDatasetV2.get_stats returns DatasetStats
# ---------------------------------------------------------------------------


def test_dataset_get_stats_type():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    pair = PreferencePair(
        prompt="p",
        chosen=Response(text="chosen text response here", score=0.9),
        rejected=Response(text="rejected short", score=0.3),
        margin=0.6,
    )
    ds.add(pair)
    stats = ds.get_stats()
    assert isinstance(stats, DatasetStats)


# ---------------------------------------------------------------------------
# 13. DatasetStats.mean_margin in valid range
# ---------------------------------------------------------------------------


def test_dataset_stats_mean_margin_valid():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    margins = [0.3, 0.5, 0.7]
    for i, m in enumerate(margins):
        pair = PreferencePair(
            prompt=f"prompt_{i}",
            chosen=Response(text=f"chosen response number {i} with decent length", score=0.5 + m),
            rejected=Response(text=f"rejected response number {i} short", score=0.5),
            margin=m,
        )
        ds.add(pair)
    stats = ds.get_stats()
    assert stats.min_margin <= stats.mean_margin <= stats.max_margin


# ---------------------------------------------------------------------------
# 14. PreferenceDatasetV2.to_dicts returns list with required keys
# ---------------------------------------------------------------------------


def test_dataset_to_dicts_keys():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    pair = PreferencePair(
        prompt="What is AI?",
        chosen=Response(text="AI is artificial intelligence.", score=0.8),
        rejected=Response(text="Dunno.", score=0.2),
        margin=0.6,
    )
    ds.add(pair)
    dicts = ds.to_dicts()
    assert isinstance(dicts, list)
    assert len(dicts) == 1
    assert "prompt" in dicts[0]
    assert "chosen" in dicts[0]
    assert "rejected" in dicts[0]


# ---------------------------------------------------------------------------
# 15. PreferenceCollectorV2.collect returns list of PreferencePair
# ---------------------------------------------------------------------------


def test_collector_collect_returns_list():
    config = CollectionConfig(
        n_candidates=4,
        min_margin=0.0,
        max_pairs_per_prompt=3,
        dedup_threshold=1.0,  # only exact duplicates filtered
    )
    collector = PreferenceCollectorV2(
        generate_fn=mock_generate,
        reward_fn=mock_reward,
        config=config,
    )
    result = collector.collect("Tell me about Python.")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, PreferencePair)


# ---------------------------------------------------------------------------
# 16. PreferenceCollectorV2.collect_batch returns PreferenceDatasetV2
# ---------------------------------------------------------------------------


def test_collector_collect_batch_returns_dataset():
    config = CollectionConfig(
        n_candidates=4,
        min_margin=0.0,
        max_pairs_per_prompt=3,
        dedup_threshold=1.0,
    )
    collector = PreferenceCollectorV2(
        generate_fn=mock_generate,
        reward_fn=mock_reward,
        config=config,
    )
    prompts = ["What is machine learning?", "Explain neural networks.", "What is PyTorch?"]
    result = collector.collect_batch(prompts)
    assert isinstance(result, PreferenceDatasetV2)


# ---------------------------------------------------------------------------
# Bonus: dataset property returns PreferenceDatasetV2
# ---------------------------------------------------------------------------


def test_collector_dataset_property():
    collector = PreferenceCollectorV2(
        generate_fn=mock_generate,
        reward_fn=mock_reward,
    )
    assert isinstance(collector.dataset, PreferenceDatasetV2)


# ---------------------------------------------------------------------------
# Bonus: filter_by_margin narrows pairs correctly
# ---------------------------------------------------------------------------


def test_dataset_filter_by_margin():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    for margin in [0.1, 0.3, 0.6, 0.9]:
        pair = PreferencePair(
            prompt="p",
            chosen=Response(text="chosen response text here is longer", score=margin + 0.5),
            rejected=Response(text="rejected response text", score=0.5),
            margin=margin,
        )
        ds.add(pair)
    assert len(ds) == 4
    filtered = ds.filter_by_margin(0.5)
    assert all(p.margin >= 0.5 for p in filtered.to_list())


# ---------------------------------------------------------------------------
# Bonus: get_stats on empty dataset returns zeros
# ---------------------------------------------------------------------------


def test_dataset_stats_empty():
    ds = PreferenceDatasetV2()
    stats = ds.get_stats()
    assert stats.n_pairs == 0
    assert stats.mean_margin == 0.0


# ---------------------------------------------------------------------------
# Bonus: sample strategies work correctly
# ---------------------------------------------------------------------------


def test_dataset_sample_high_margin():
    config = CollectionConfig(min_margin=0.0)
    ds = PreferenceDatasetV2(config=config)
    for margin in [0.1, 0.5, 0.9]:
        pair = PreferencePair(
            prompt="p",
            chosen=Response(text="chosen response text long enough here", score=margin + 0.5),
            rejected=Response(text="rejected response text", score=0.5),
            margin=margin,
        )
        ds.add(pair)
    sampled = ds.sample(2, strategy="high_margin")
    assert len(sampled) == 2
    # Should be top-2 by margin: 0.9, 0.5
    margins = sorted([p.margin for p in sampled], reverse=True)
    assert margins[0] == pytest.approx(0.9)
    assert margins[1] == pytest.approx(0.5)
