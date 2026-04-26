"""Tests for src/data/synthetic_preference.py."""

from __future__ import annotations

import random

from src.data.synthetic_preference import (
    PreferenceDataConfig,
    PreferencePair,
    SyntheticPreferenceGenerator,
    augment_with_critique,
    filter_by_quality,
    format_for_dpo,
    format_for_rlhf,
    generate_coding_pair,
    generate_math_pair,
    generate_reasoning_pair,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# PreferenceDataConfig defaults
# ---------------------------------------------------------------------------


def test_preference_data_config_defaults():
    cfg = PreferenceDataConfig()
    assert cfg.n_pairs == 100
    assert cfg.seed == 42
    assert cfg.augment_with_critique is True
    assert cfg.quality_threshold == 0.5
    assert cfg.domains == ["math", "coding", "reasoning"]


# ---------------------------------------------------------------------------
# generate_math_pair
# ---------------------------------------------------------------------------


def test_generate_math_pair_returns_preference_pair():
    pair = generate_math_pair(_rng())
    assert isinstance(pair, PreferencePair)


def test_generate_math_pair_domain():
    pair = generate_math_pair(_rng())
    assert pair.domain == "math"


def test_generate_math_pair_chosen_ne_rejected():
    pair = generate_math_pair(_rng())
    assert pair.chosen != pair.rejected


def test_generate_math_pair_has_prompt():
    pair = generate_math_pair(_rng())
    assert isinstance(pair.prompt, str)
    assert len(pair.prompt) > 0


# ---------------------------------------------------------------------------
# generate_coding_pair
# ---------------------------------------------------------------------------


def test_generate_coding_pair_domain():
    pair = generate_coding_pair(_rng())
    assert pair.domain == "coding"


def test_generate_coding_pair_returns_preference_pair():
    pair = generate_coding_pair(_rng())
    assert isinstance(pair, PreferencePair)


def test_generate_coding_pair_chosen_ne_rejected():
    pair = generate_coding_pair(_rng())
    assert pair.chosen != pair.rejected


# ---------------------------------------------------------------------------
# generate_reasoning_pair
# ---------------------------------------------------------------------------


def test_generate_reasoning_pair_domain():
    pair = generate_reasoning_pair(_rng())
    assert pair.domain == "reasoning"


def test_generate_reasoning_pair_returns_preference_pair():
    pair = generate_reasoning_pair(_rng())
    assert isinstance(pair, PreferencePair)


def test_generate_reasoning_pair_chosen_ne_rejected():
    pair = generate_reasoning_pair(_rng())
    assert pair.chosen != pair.rejected


# ---------------------------------------------------------------------------
# augment_with_critique
# ---------------------------------------------------------------------------


def test_augment_with_critique_adds_critique():
    pair = generate_math_pair(_rng())
    assert pair.critique is None
    augmented = augment_with_critique(pair, _rng())
    assert augmented.critique is not None
    assert isinstance(augmented.critique, str)
    assert len(augmented.critique) > 0


def test_augment_with_critique_preserves_fields():
    pair = generate_math_pair(_rng(7))
    augmented = augment_with_critique(pair, _rng())
    assert augmented.prompt == pair.prompt
    assert augmented.chosen == pair.chosen
    assert augmented.rejected == pair.rejected
    assert augmented.domain == pair.domain


def test_augment_with_critique_template_format():
    pair = generate_math_pair(_rng())
    augmented = augment_with_critique(pair, _rng())
    assert "chosen response" in augmented.critique
    assert "rejected response" in augmented.critique


# ---------------------------------------------------------------------------
# filter_by_quality
# ---------------------------------------------------------------------------


def test_filter_by_quality_removes_low_margin():
    pairs = [
        PreferencePair("p", "c", "r", "math", score_chosen=0.6, score_rejected=0.5),  # diff=0.1
        PreferencePair("p", "c", "r", "math", score_chosen=1.0, score_rejected=0.0),  # diff=1.0
    ]
    result = filter_by_quality(pairs, threshold=0.5)
    assert len(result) == 1
    assert result[0].score_chosen == 1.0


def test_filter_by_quality_threshold_zero_all_pass():
    pairs = [
        PreferencePair("p", "c", "r", "math", score_chosen=0.6, score_rejected=0.5),
        PreferencePair("p", "c", "r", "math", score_chosen=1.0, score_rejected=0.0),
        PreferencePair("p", "c", "r", "coding", score_chosen=0.3, score_rejected=0.3),
    ]
    result = filter_by_quality(pairs, threshold=0.0)
    assert len(result) == 3


def test_filter_by_quality_empty_input():
    assert filter_by_quality([], threshold=0.5) == []


# ---------------------------------------------------------------------------
# format_for_dpo
# ---------------------------------------------------------------------------


def test_format_for_dpo_keys_present():
    pair = generate_math_pair(_rng())
    result = format_for_dpo(pair)
    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result


def test_format_for_dpo_values_match():
    pair = generate_math_pair(_rng(3))
    result = format_for_dpo(pair)
    assert result["prompt"] == pair.prompt
    assert result["chosen"] == pair.chosen
    assert result["rejected"] == pair.rejected


# ---------------------------------------------------------------------------
# format_for_rlhf
# ---------------------------------------------------------------------------


def test_format_for_rlhf_keys_present():
    pair = generate_math_pair(_rng())
    result = format_for_rlhf(pair)
    assert "prompt" in result
    assert "completion" in result
    assert "reward" in result


def test_format_for_rlhf_uses_chosen_and_score():
    pair = generate_coding_pair(_rng(5))
    result = format_for_rlhf(pair)
    assert result["completion"] == pair.chosen
    assert result["reward"] == pair.score_chosen


# ---------------------------------------------------------------------------
# SyntheticPreferenceGenerator
# ---------------------------------------------------------------------------


def test_generator_generate_returns_list():
    cfg = PreferenceDataConfig(n_pairs=10, augment_with_critique=False, quality_threshold=0.0)
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate()
    assert isinstance(result, list)


def test_generator_generate_returns_preference_pair_objects():
    cfg = PreferenceDataConfig(n_pairs=10, augment_with_critique=False, quality_threshold=0.0)
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate()
    assert all(isinstance(p, PreferencePair) for p in result)


def test_generator_generate_n_overrides_config():
    cfg = PreferenceDataConfig(n_pairs=100, augment_with_critique=False, quality_threshold=0.0)
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate(n=5)
    # After quality filter with threshold=0, all 5 pairs should survive
    assert len(result) == 5


def test_generator_generate_with_critique():
    cfg = PreferenceDataConfig(n_pairs=6, augment_with_critique=True, quality_threshold=0.0)
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate()
    assert all(p.critique is not None for p in result)


def test_generator_generate_distributes_across_domains():
    cfg = PreferenceDataConfig(
        n_pairs=9,
        augment_with_critique=False,
        quality_threshold=0.0,
        domains=["math", "coding", "reasoning"],
    )
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate()
    domains = {p.domain for p in result}
    assert "math" in domains
    assert "coding" in domains
    assert "reasoning" in domains


def test_generator_reproducible_with_same_seed():
    cfg1 = PreferenceDataConfig(
        n_pairs=5, seed=99, augment_with_critique=False, quality_threshold=0.0
    )
    cfg2 = PreferenceDataConfig(
        n_pairs=5, seed=99, augment_with_critique=False, quality_threshold=0.0
    )
    result1 = SyntheticPreferenceGenerator(cfg1).generate()
    result2 = SyntheticPreferenceGenerator(cfg2).generate()
    assert [p.prompt for p in result1] == [p.prompt for p in result2]


def test_generator_unknown_domain_skipped():
    cfg = PreferenceDataConfig(
        n_pairs=4,
        augment_with_critique=False,
        quality_threshold=0.0,
        domains=["math", "unknown_domain"],
    )
    gen = SyntheticPreferenceGenerator(cfg)
    result = gen.generate()
    # unknown_domain is skipped; all pairs come from math
    assert all(p.domain == "math" for p in result)
