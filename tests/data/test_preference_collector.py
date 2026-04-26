"""Tests for src/data/preference_collector.py — preference pair collection for DPO/RLHF."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.data.preference_collector import (
    PreferenceCollector,
    PreferenceConfig,
    PreferencePair,
    ResponsePool,
    score_response_heuristic,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_model(vocab_size: int = 256, seq_len_ignored: bool = True):
    """Return a mock model that outputs uniform logits."""
    model = MagicMock()

    def _forward(input_ids):
        batch, seq = input_ids.shape
        logits = torch.ones(batch, seq, vocab_size)
        return None, logits, None

    model.side_effect = _forward
    return model


def _encode(text: str) -> list[int]:
    return [ord(c) % 256 for c in text]


def _decode(ids: list[int]) -> str:
    return "".join(chr(i + 32) for i in ids)


def _make_collector(score_fn=None, n_responses: int = 2):
    cfg = PreferenceConfig(n_responses=n_responses, max_length=5)
    model = _make_model()
    return PreferenceCollector(
        model=model,
        config=cfg,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        score_fn=score_fn,
    )


# ---------------------------------------------------------------------------
# 1. PreferenceConfig defaults
# ---------------------------------------------------------------------------


def test_preference_config_defaults():
    cfg = PreferenceConfig()
    assert cfg.n_responses == 4
    assert cfg.temperature == 0.8
    assert cfg.diversity_penalty == 0.1
    assert cfg.min_length == 10
    assert cfg.max_length == 256


# ---------------------------------------------------------------------------
# 2. PreferencePair fields and margin == score_chosen - score_rejected
# ---------------------------------------------------------------------------


def test_preference_pair_fields_and_margin():
    pair = PreferencePair(
        prompt="Hello?",
        chosen="Great response.",
        rejected="Bad response.",
        score_chosen=0.9,
        score_rejected=0.3,
        margin=0.6,
    )
    assert pair.prompt == "Hello?"
    assert pair.chosen == "Great response."
    assert pair.rejected == "Bad response."
    assert pair.score_chosen == pytest.approx(0.9)
    assert pair.score_rejected == pytest.approx(0.3)
    assert pair.margin == pytest.approx(pair.score_chosen - pair.score_rejected)
    assert pair.source == "auto"


# ---------------------------------------------------------------------------
# 3. ResponsePool.add increases length
# ---------------------------------------------------------------------------


def test_response_pool_add_increases_length():
    pool = ResponsePool()
    assert len(pool) == 0
    pool.add("text one", 0.5)
    assert len(pool) == 1
    pool.add("text two", 0.8)
    assert len(pool) == 2


# ---------------------------------------------------------------------------
# 4. ResponsePool.get_best_worst returns highest/lowest score texts
# ---------------------------------------------------------------------------


def test_response_pool_get_best_worst():
    pool = ResponsePool()
    pool.add("low", 0.1)
    pool.add("mid", 0.5)
    pool.add("high", 0.9)

    best, worst = pool.get_best_worst()
    assert best == "high"
    assert worst == "low"


# ---------------------------------------------------------------------------
# 5. ResponsePool.get_best_worst single item returns same for both
# ---------------------------------------------------------------------------


def test_response_pool_get_best_worst_single_item():
    pool = ResponsePool()
    pool.add("only response", 0.7)
    best, worst = pool.get_best_worst()
    assert best == worst == "only response"


# ---------------------------------------------------------------------------
# 6. score_response_heuristic — longer response scores higher than very short
# ---------------------------------------------------------------------------


def test_score_heuristic_longer_response_higher():
    prompt = "Tell me about Python."
    short_response = "Python."
    long_response = (
        "Python is a high-level, general-purpose programming language. "
        "It is widely used for web development, data science, and automation. "
        "Python emphasizes code readability and simplicity."
    )
    short_score = score_response_heuristic(prompt, short_response)
    long_score = score_response_heuristic(prompt, long_response)
    assert long_score > short_score


# ---------------------------------------------------------------------------
# 7. score_response_heuristic — prompt-relevant response scores higher
# ---------------------------------------------------------------------------


def test_score_heuristic_relevant_response_higher():
    prompt = "What is Python?"
    relevant = "Python is a popular programming language used in many domains."
    irrelevant = "The weather today is sunny and warm outside in the park."
    assert score_response_heuristic(prompt, relevant) > score_response_heuristic(prompt, irrelevant)


# ---------------------------------------------------------------------------
# 8. score_response_heuristic — returns value in [0, 1]
# ---------------------------------------------------------------------------


def test_score_heuristic_in_range():
    prompt = "Hello world."
    responses = [
        "",
        "Hi.",
        "This is a moderately long response that covers the topic well!",
        "x" * 1000,
    ]
    for r in responses:
        s = score_response_heuristic(prompt, r)
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for response of length {len(r)}"


# ---------------------------------------------------------------------------
# 9. PreferenceCollector.collect_for_prompt returns PreferencePair or None
# ---------------------------------------------------------------------------


def test_collector_collect_for_prompt_returns_pair_or_none():
    call_count = [0]
    responses = [
        ("Short.", 0.2),
        ("A much longer and more relevant response to the prompt here!", 0.8),
    ]

    def score_fn(prompt, response):
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return responses[idx][1]

    cfg = PreferenceConfig(n_responses=2, max_length=5)
    model = _make_model()
    collector = PreferenceCollector(
        model=model,
        config=cfg,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        score_fn=score_fn,
    )
    result = collector.collect_for_prompt("Tell me something.")
    # Result is either PreferencePair or None
    assert result is None or isinstance(result, PreferencePair)


# ---------------------------------------------------------------------------
# 10. PreferenceCollector.collect_dataset returns list
# ---------------------------------------------------------------------------


def test_collector_collect_dataset_returns_list():
    collector = _make_collector()
    prompts = ["What is AI?", "Explain gravity."]
    result = collector.collect_dataset(prompts)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 11. PreferenceCollector.filter_pairs removes low-margin pairs
# ---------------------------------------------------------------------------


def test_collector_filter_pairs_removes_low_margin():
    pairs = [
        PreferencePair("p1", "c1", "r1", 0.9, 0.8, 0.1),  # margin exactly 0.1 — kept
        PreferencePair("p2", "c2", "r2", 0.9, 0.85, 0.05),  # margin 0.05 — filtered
        PreferencePair("p3", "c3", "r3", 0.9, 0.5, 0.4),  # margin 0.4 — kept
    ]
    collector = _make_collector()
    filtered = collector.filter_pairs(pairs, min_margin=0.1)
    assert len(filtered) == 2
    assert all(p.margin >= 0.1 for p in filtered)


# ---------------------------------------------------------------------------
# 12. PreferenceCollector.export_dpo_format has required keys
# ---------------------------------------------------------------------------


def test_collector_export_dpo_format_keys():
    pairs = [
        PreferencePair("prompt A", "chosen A", "rejected A", 0.9, 0.3, 0.6),
        PreferencePair("prompt B", "chosen B", "rejected B", 0.8, 0.4, 0.4),
    ]
    collector = _make_collector()
    records = collector.export_dpo_format(pairs)
    assert len(records) == 2
    for rec in records:
        assert "prompt" in rec
        assert "chosen" in rec
        assert "rejected" in rec


# ---------------------------------------------------------------------------
# 13. ResponsePool.get_diverse_pair — chosen has higher score than rejected
# ---------------------------------------------------------------------------


def test_response_pool_get_diverse_pair_chosen_higher_score():
    pool = ResponsePool()
    pool.add("aaaa bbbb cccc dddd eeee", 0.9)
    pool.add("xxxx yyyy zzzz wwww vvvv", 0.3)
    pool.add("1111 2222 3333 4444 5555", 0.1)

    def diversity_fn(a: str, b: str) -> float:
        # Jaccard distance on characters
        set_a = set(a)
        set_b = set(b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return 1.0 - (inter / union if union else 0.0)

    chosen, rejected = pool.get_diverse_pair(diversity_fn)

    # chosen must be the highest-scored response
    chosen_score = next(s for t, s in pool._responses if t == chosen)
    rejected_score = next(s for t, s in pool._responses if t == rejected)
    assert chosen_score > rejected_score


# ---------------------------------------------------------------------------
# 14. PreferenceCollector.collect_dataset length <= len(prompts)
# ---------------------------------------------------------------------------


def test_collector_collect_dataset_length_le_prompts():
    collector = _make_collector()
    prompts = ["Q1?", "Q2?", "Q3?"]
    result = collector.collect_dataset(prompts)
    assert len(result) <= len(prompts)


# ---------------------------------------------------------------------------
# 15. export_dpo_format each dict has prompt/chosen/rejected keys
# ---------------------------------------------------------------------------


def test_export_dpo_format_each_dict_has_all_keys():
    collector = _make_collector()
    pairs = [
        PreferencePair("p", "c", "r", 0.9, 0.1, 0.8, source="test"),
    ]
    records = collector.export_dpo_format(pairs)
    assert len(records) == 1
    rec = records[0]
    assert set(rec.keys()) >= {"prompt", "chosen", "rejected"}
    assert rec["prompt"] == "p"
    assert rec["chosen"] == "c"
    assert rec["rejected"] == "r"
