"""Unit tests for src.eval.multi_needle_eval.

Pure-stdlib tests; no foreign imports. Exercises config validation, sample
construction, depth-profile correctness, tolerant recovery parsing, and
scoring semantics for the multi-needle NIAH harness.
"""

from __future__ import annotations

import random

import pytest

from src.eval.multi_needle_eval import (
    DEPTH_PROFILE_REGISTRY,
    MULTI_NEEDLE_REGISTRY,
    MultiNeedleConfig,
    MultiNeedleError,
    MultiNeedleSample,
    MultiNeedleVerdict,
    build_sample,
    parse_recovery,
    register_depth_profile,
    score,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_good_config_construction():
    cfg = MultiNeedleConfig(
        num_needles=4, haystack_tokens=256, depth_profile="uniform", seed=7
    )
    assert cfg.num_needles == 4
    assert cfg.haystack_tokens == 256
    assert cfg.depth_profile == "uniform"
    assert cfg.seed == 7
    assert cfg.value_digits == 6


def test_bad_num_needles_too_small_raises():
    cfg = MultiNeedleConfig(num_needles=1, haystack_tokens=128, depth_profile="uniform")
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


def test_bad_num_needles_too_large_raises():
    cfg = MultiNeedleConfig(num_needles=17, haystack_tokens=128, depth_profile="uniform")
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


def test_bad_haystack_tokens_too_small_raises():
    cfg = MultiNeedleConfig(num_needles=4, haystack_tokens=63, depth_profile="uniform")
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


def test_bad_haystack_tokens_too_large_raises():
    cfg = MultiNeedleConfig(
        num_needles=4, haystack_tokens=65537, depth_profile="uniform"
    )
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


def test_bad_depth_profile_raises():
    cfg = MultiNeedleConfig(
        num_needles=4, haystack_tokens=128, depth_profile="spiral_of_doom"
    )
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


def test_bad_value_digits_raises():
    cfg = MultiNeedleConfig(
        num_needles=4,
        haystack_tokens=128,
        depth_profile="uniform",
        value_digits=1,
    )
    with pytest.raises(MultiNeedleError):
        build_sample(cfg)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
def test_build_sample_returns_k_needles():
    cfg = MultiNeedleConfig(num_needles=5, haystack_tokens=512, depth_profile="uniform")
    s = build_sample(cfg)
    assert isinstance(s, MultiNeedleSample)
    assert len(s.needles) == 5
    assert len(s.gold) == 5
    assert len(s.depth_fracs) == 5


def test_all_keys_unique():
    cfg = MultiNeedleConfig(num_needles=8, haystack_tokens=1024, depth_profile="uniform")
    s = build_sample(cfg)
    keys = [k for (k, _) in s.gold]
    assert len(set(keys)) == len(keys) == 8


def test_prompt_contains_every_needle_text_exactly_once():
    cfg = MultiNeedleConfig(num_needles=6, haystack_tokens=1024, depth_profile="uniform")
    s = build_sample(cfg)
    for key, value in s.gold:
        needle_text = f"remember the code {key}={value}."
        assert s.prompt.count(needle_text) == 1, (
            f"needle {key}={value} must appear exactly once; "
            f"got {s.prompt.count(needle_text)}"
        )


def test_prompt_length_roughly_matches_haystack_tokens():
    ht = 2048
    cfg = MultiNeedleConfig(num_needles=4, haystack_tokens=ht, depth_profile="uniform")
    s = build_sample(cfg)
    n_words = len(s.prompt.split())
    # ±30% tolerance as specified.
    assert ht * 0.7 <= n_words <= ht * 1.3


def test_determinism_same_seed_byte_identical():
    cfg = MultiNeedleConfig(
        num_needles=4, haystack_tokens=512, depth_profile="uniform", seed=42
    )
    s1 = build_sample(cfg)
    s2 = build_sample(cfg)
    assert s1.prompt == s2.prompt
    assert s1.gold == s2.gold
    assert s1.depth_fracs == s2.depth_fracs
    assert s1.needles == s2.needles


def test_different_seeds_produce_different_samples():
    cfg_a = MultiNeedleConfig(
        num_needles=4, haystack_tokens=512, depth_profile="uniform", seed=1
    )
    cfg_b = MultiNeedleConfig(
        num_needles=4, haystack_tokens=512, depth_profile="uniform", seed=2
    )
    s_a = build_sample(cfg_a)
    s_b = build_sample(cfg_b)
    assert s_a.prompt != s_b.prompt
    assert s_a.gold != s_b.gold


# ---------------------------------------------------------------------------
# Depth profile correctness
# ---------------------------------------------------------------------------
def test_uniform_depth_fracs_approximately_evenly_spaced():
    cfg = MultiNeedleConfig(num_needles=5, haystack_tokens=512, depth_profile="uniform")
    s = build_sample(cfg)
    depths = s.depth_fracs
    assert depths == tuple(sorted(depths))
    # Consecutive differences should be approximately equal (constant step).
    diffs = [depths[i + 1] - depths[i] for i in range(len(depths) - 1)]
    assert all(abs(d - diffs[0]) < 1e-9 for d in diffs)


def test_clustered_early_all_depths_below_0_2():
    cfg = MultiNeedleConfig(
        num_needles=6, haystack_tokens=1024, depth_profile="clustered_early"
    )
    s = build_sample(cfg)
    assert all(d < 0.2 for d in s.depth_fracs)


def test_clustered_late_all_depths_above_0_8():
    cfg = MultiNeedleConfig(
        num_needles=6, haystack_tokens=1024, depth_profile="clustered_late"
    )
    s = build_sample(cfg)
    assert all(d > 0.8 for d in s.depth_fracs)


def test_boundary_half_early_half_late():
    cfg = MultiNeedleConfig(
        num_needles=8, haystack_tokens=1024, depth_profile="boundary"
    )
    s = build_sample(cfg)
    lows = [d for d in s.depth_fracs if d < 0.1]
    highs = [d for d in s.depth_fracs if d > 0.9]
    assert len(lows) == 4 and len(highs) == 4
    assert len(lows) + len(highs) == 8


# ---------------------------------------------------------------------------
# parse_recovery
# ---------------------------------------------------------------------------
def test_parse_recovery_basic_pairs():
    got = parse_recovery("AB=123456\nCD=654321\n")
    assert got == (("AB", "123456"), ("CD", "654321"))


def test_parse_recovery_tolerates_bullets():
    got = parse_recovery("- AB=123456\n* CD=654321")
    assert got == (("AB", "123456"), ("CD", "654321"))


def test_parse_recovery_dedupes_first_wins():
    got = parse_recovery("AB=111111\nAB=222222")
    assert got == (("AB", "111111"),)


def test_parse_recovery_rejects_malformed():
    assert parse_recovery("AB==123") == ()
    assert parse_recovery("=123") == ()
    assert parse_recovery("AB=") == ()
    assert parse_recovery("ab=123") == ()


def test_parse_recovery_empty_input():
    assert parse_recovery("") == ()


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def _make_small_sample() -> MultiNeedleSample:
    cfg = MultiNeedleConfig(
        num_needles=4, haystack_tokens=512, depth_profile="uniform", seed=0
    )
    return build_sample(cfg)


def test_score_perfect_output_all_or_nothing_true():
    s = _make_small_sample()
    lines = "\n".join(f"{k}={v}" for (k, v) in sorted(s.gold))
    v = score(s, lines)
    assert isinstance(v, MultiNeedleVerdict)
    assert v.recall_exact == 1.0
    assert v.recall_key == 1.0
    assert v.precision == 1.0
    assert v.all_or_nothing is True


def test_score_empty_output_yields_zero_recall_unit_precision():
    s = _make_small_sample()
    v = score(s, "")
    assert v.recall_exact == 0.0
    assert v.recall_key == 0.0
    assert v.precision == 1.0
    assert v.all_or_nothing is False


def test_score_partial_3_of_4_correct_gives_0_75():
    s = _make_small_sample()
    first3 = list(s.gold)[:3]
    lines = "\n".join(f"{k}={v}" for (k, v) in first3)
    v = score(s, lines)
    assert v.recall_exact == 0.75
    assert v.recall_key == 0.75


def test_score_wrong_value_counts_as_key_hit_only():
    s = _make_small_sample()
    # Take first gold key but corrupt its value.
    (k0, _) = s.gold[0]
    rest = s.gold[1:]
    wrong = [(k0, "000000")] + list(rest)
    lines = "\n".join(f"{a}={b}" for (a, b) in wrong)
    v = score(s, lines)
    # 3 of 4 pairs exact, but all 4 keys present.
    assert v.recall_exact == 0.75
    assert v.recall_key == 1.0
    assert v.all_or_nothing is False


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
def test_depth_profile_registry_contains_four_profiles():
    for name in ("uniform", "clustered_early", "clustered_late", "boundary"):
        assert name in DEPTH_PROFILE_REGISTRY
        assert callable(DEPTH_PROFILE_REGISTRY[name])


def test_multi_needle_registry_contains_three_reference_configs():
    assert "niah-mk-small" in MULTI_NEEDLE_REGISTRY
    assert "niah-mk-medium" in MULTI_NEEDLE_REGISTRY
    assert "niah-mk-boundary" in MULTI_NEEDLE_REGISTRY
    small = MULTI_NEEDLE_REGISTRY["niah-mk-small"]
    med = MULTI_NEEDLE_REGISTRY["niah-mk-medium"]
    bound = MULTI_NEEDLE_REGISTRY["niah-mk-boundary"]
    assert small.num_needles == 4 and small.haystack_tokens == 512
    assert small.depth_profile == "uniform"
    assert med.num_needles == 8 and med.haystack_tokens == 4096
    assert med.depth_profile == "uniform"
    assert bound.num_needles == 8 and bound.haystack_tokens == 4096
    assert bound.depth_profile == "boundary"


def test_register_depth_profile_adds_custom_and_builds_sample():
    def center_only(k: int, rng: random.Random) -> tuple:
        del rng
        return tuple(0.5 for _ in range(k))

    register_depth_profile("center_only", center_only)
    assert "center_only" in DEPTH_PROFILE_REGISTRY

    cfg = MultiNeedleConfig(
        num_needles=3, haystack_tokens=256, depth_profile="center_only", seed=3
    )
    s = build_sample(cfg)
    assert len(s.gold) == 3
    assert all(abs(d - 0.5) < 1e-12 for d in s.depth_fracs)
    # All three needle texts must appear in the prompt.
    for key, value in s.gold:
        assert f"remember the code {key}={value}." in s.prompt
