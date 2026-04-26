"""Integration tests for multi_needle_eval through the src.eval package surface."""

from __future__ import annotations

import src.eval as eval_pkg
from src.eval import (
    BENCHMARK_REGISTRY,
    MULTI_NEEDLE_REGISTRY,
    MultiNeedleConfig,
    multi_needle_build_sample,
    multi_needle_score,
)


def test_package_level_imports_work():
    # Also confirm a few collateral symbols re-exported correctly.
    assert callable(multi_needle_build_sample)
    assert callable(multi_needle_score)
    assert MultiNeedleConfig is not None
    assert hasattr(eval_pkg, "multi_needle_parse_recovery")
    assert hasattr(eval_pkg, "MultiNeedleError")
    assert hasattr(eval_pkg, "MultiNeedleSample")
    assert hasattr(eval_pkg, "MultiNeedleVerdict")
    assert hasattr(eval_pkg, "DEPTH_PROFILE_REGISTRY")
    assert hasattr(eval_pkg, "register_depth_profile")


def test_benchmark_registry_contains_niah_mk():
    assert "niah_mk" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["niah_mk"] is MultiNeedleConfig
    # Preexisting entries must remain untouched.
    assert "niah" in BENCHMARK_REGISTRY
    assert "ruler" in BENCHMARK_REGISTRY


def test_end_to_end_perfect_recovery_all_or_nothing():
    cfg = MultiNeedleConfig(num_needles=4, haystack_tokens=512, depth_profile="uniform", seed=11)
    s = multi_needle_build_sample(cfg)
    lines = "\n".join(f"{k}={v}" for (k, v) in sorted(s.gold))
    v = multi_needle_score(s, lines)
    assert v.all_or_nothing is True
    assert v.recall_exact == 1.0
    assert v.precision == 1.0


def test_registry_round_trip_perfect_score():
    cfg = MULTI_NEEDLE_REGISTRY["niah-mk-small"]
    s = multi_needle_build_sample(cfg)
    assert len(s.gold) == cfg.num_needles
    lines = "\n".join(f"{k}={v}" for (k, v) in sorted(s.gold))
    v = multi_needle_score(s, lines)
    assert v.recall_exact == 1.0
    assert v.all_or_nothing is True


def test_three_seeds_produce_pairwise_distinct_prompts():
    base = MULTI_NEEDLE_REGISTRY["niah-mk-small"]
    prompts = []
    for seed in (1, 2, 3):
        cfg = MultiNeedleConfig(
            num_needles=base.num_needles,
            haystack_tokens=base.haystack_tokens,
            depth_profile=base.depth_profile,
            seed=seed,
        )
        prompts.append(multi_needle_build_sample(cfg).prompt)
    assert prompts[0] != prompts[1]
    assert prompts[1] != prompts[2]
    assert prompts[0] != prompts[2]
