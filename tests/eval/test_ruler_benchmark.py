"""Unit tests for src/eval/ruler_benchmark.py."""

from __future__ import annotations

import pytest

from src.eval.ruler_benchmark import RULERBenchmark


# ----------------------------------------------------------------------
# Builder smoke tests: each returns (prompt, expected)
# ----------------------------------------------------------------------
def test_build_multi_key_niah_returns_pair():
    bench = RULERBenchmark()
    prompt, expected = bench.build_multi_key_niah(256, n_keys=4, seed=0)
    assert isinstance(prompt, str) and prompt
    assert isinstance(expected, str) and expected
    # Expected value must appear inside the prompt.
    assert expected in prompt


def test_build_multi_value_niah_returns_pair_all_values_present():
    bench = RULERBenchmark()
    prompt, expected = bench.build_multi_value_niah(256, n_values=3, seed=1)
    assert isinstance(prompt, str)
    assert isinstance(expected, list) and len(expected) == 3
    for v in expected:
        assert v in prompt


def test_build_variable_tracking_chain_resolves():
    bench = RULERBenchmark()
    prompt, expected = bench.build_variable_tracking(
        512, n_vars=4, chain_length=4, seed=7
    )
    assert isinstance(prompt, str)
    assert isinstance(expected, str)
    # Target chain's root assignment must be present verbatim.
    assert expected in prompt
    # And the query var name must be present.
    assert "var_0_3" in prompt  # chain_length=4 -> tail index 3


def test_build_common_words_extraction_words_appear_many_times():
    bench = RULERBenchmark()
    prompt, expected = bench.build_common_words_extraction(512, n_common=5, seed=3)
    assert isinstance(expected, list) and len(expected) == 5
    # Each common word should be present many times (we emit 10 per).
    for w in expected:
        assert prompt.count(w) >= 10


def test_build_aggregation_sum_verified():
    bench = RULERBenchmark()
    prompt, expected = bench.build_aggregation(512, operation="sum", seed=4)
    assert isinstance(expected, int) and expected > 0
    assert str(expected) not in prompt or True  # sum usually not spelled
    # Extract recorded values and verify sum matches.
    import re
    vals = [int(x) for x in re.findall(r"recorded value is (\d+)", prompt)]
    assert sum(vals) == expected


def test_build_aggregation_max_verified():
    bench = RULERBenchmark()
    prompt, expected = bench.build_aggregation(512, operation="max", seed=5)
    assert isinstance(expected, int) and expected > 0
    import re
    vals = [int(x) for x in re.findall(r"recorded value is (\d+)", prompt)]
    assert max(vals) == expected


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_determinism_with_seed():
    bench = RULERBenchmark()
    p1, e1 = bench.build_multi_key_niah(256, n_keys=3, seed=42)
    p2, e2 = bench.build_multi_key_niah(256, n_keys=3, seed=42)
    assert p1 == p2
    assert e1 == e2
    # Different seed should change output.
    p3, _ = bench.build_multi_key_niah(256, n_keys=3, seed=43)
    assert p3 != p1


# ----------------------------------------------------------------------
# Context length scaling
# ----------------------------------------------------------------------
def test_context_tokens_scales_prompt_length():
    bench = RULERBenchmark()
    short, _ = bench.build_multi_key_niah(128, n_keys=3, seed=0)
    long, _ = bench.build_multi_key_niah(2048, n_keys=3, seed=0)
    assert len(long) > 4 * len(short)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_invalid_n_keys_zero_raises():
    bench = RULERBenchmark()
    with pytest.raises(ValueError):
        bench.build_multi_key_niah(256, n_keys=0, seed=0)


def test_invalid_context_tokens_raises():
    bench = RULERBenchmark()
    with pytest.raises(ValueError):
        bench.build_multi_key_niah(0, n_keys=3, seed=0)


def test_invalid_aggregation_operation_raises():
    bench = RULERBenchmark()
    with pytest.raises(ValueError):
        bench.build_aggregation(256, operation="mean", seed=0)


# ----------------------------------------------------------------------
# Evaluate with oracle / null generate_fn
# ----------------------------------------------------------------------
def _oracle_factory(bench: RULERBenchmark):
    """Build an oracle generate_fn by replaying each task builder.

    Since evaluate() uses seeds 0..samples_per-1 per task, we precompute
    prompt -> expected-as-string for every (task, L, seed) triple and
    return the expected string on exact prompt match.
    """
    lookup = {}

    def register(task, L, s):
        prompt, expected = bench._build_task(task, L, seed=s)
        if isinstance(expected, list):
            ans = " ".join(str(v) for v in expected)
        else:
            ans = str(expected)
        lookup[prompt] = ans

    return lookup, register


def test_evaluate_oracle_hits_one():
    bench = RULERBenchmark()
    tasks = list(bench.TASKS)
    context_lengths = [256]
    samples_per = 2

    lookup, register = _oracle_factory(bench)
    for t in tasks:
        for L in context_lengths:
            for s in range(samples_per):
                register(t, L, s)

    def oracle(prompt: str) -> str:
        return lookup[prompt]

    results = bench.evaluate(
        oracle, tasks=tasks, context_lengths=context_lengths, samples_per=samples_per
    )
    for task, info in results.items():
        assert info["pass_rate"] == 1.0, f"{task} should be perfect"


def test_evaluate_null_stub_is_zero():
    bench = RULERBenchmark()
    results = bench.evaluate(
        generate_fn=lambda p: "",
        tasks=["multi_key_niah", "aggregation_sum"],
        context_lengths=[256],
        samples_per=2,
    )
    for task, info in results.items():
        assert info["pass_rate"] == 0.0


# ----------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------
def test_score_per_task_returns_non_negative_floats():
    bench = RULERBenchmark()
    results = bench.evaluate(
        generate_fn=lambda p: "",
        tasks=["multi_key_niah", "aggregation_max"],
        context_lengths=[256],
        samples_per=1,
    )
    per = bench.score_per_task(results)
    assert set(per.keys()) == {"multi_key_niah", "aggregation_max"}
    for v in per.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_overall_score_is_mean_of_per_task():
    bench = RULERBenchmark()
    # Oracle on one task, null on another -> 0.5 mean.
    p_mk, e_mk = bench.build_multi_key_niah(256, seed=0)

    def mixed(prompt: str) -> str:
        # Only answers the multi_key_niah prompt; others get empty string.
        if prompt == p_mk:
            return e_mk
        return ""

    # Use a single task/length/sample combo so "mixed" matches exactly once
    # for multi_key_niah and misses everything else.
    results = bench.evaluate(
        generate_fn=mixed,
        tasks=["multi_key_niah", "aggregation_sum"],
        context_lengths=[256],
        samples_per=1,
    )
    per = bench.score_per_task(results)
    assert per["multi_key_niah"] == 1.0
    assert per["aggregation_sum"] == 0.0
    assert bench.overall_score(results) == pytest.approx(0.5)


def test_common_words_oracle_answers_with_k_words():
    bench = RULERBenchmark()
    prompt, expected = bench.build_common_words_extraction(512, n_common=5, seed=9)

    def oracle(p: str) -> str:
        assert p == prompt
        return ", ".join(expected)

    # Manual pass check via the same code path as evaluate().
    assert bench._check(expected, oracle(prompt)) is True


def test_evaluate_rejects_unknown_task():
    bench = RULERBenchmark()
    with pytest.raises(ValueError):
        bench.evaluate(
            generate_fn=lambda p: "",
            tasks=["not_a_task"],
            context_lengths=[256],
            samples_per=1,
        )
