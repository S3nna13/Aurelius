"""Unit tests for the Agent Red Teaming (ART) benchmark."""

from __future__ import annotations

import pytest

from src.eval.agent_red_team_bench import (
    DEFAULT_ATTACKER_PROFILES,
    ARTBenchmark,
    ARTResult,
    AttackAttempt,
    AttackerProfile,
    stub_judge_fn,
    stub_target_fn,
)


def _profile(pid: str, prompts):
    atts = tuple(AttackAttempt(p, "", False, 0.0, "") for p in prompts)
    return AttackerProfile(id=pid, name=pid, style="test", attempts=atts)


def _all_unsafe_profile(pid: str, n: int) -> AttackerProfile:
    return _profile(pid, [f"UNSAFE:{i}" for i in range(n)])


def _all_safe_profile(pid: str, n: int) -> AttackerProfile:
    return _profile(pid, [f"safe-{i}" for i in range(n)])


def _mixed_profile(pid: str, n_unsafe: int, n_safe: int) -> AttackerProfile:
    prompts = [f"UNSAFE:{i}" for i in range(n_unsafe)] + [f"safe-{i}" for i in range(n_safe)]
    return _profile(pid, prompts)


def test_k_values_default_is_1_10_100():
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn)
    assert bench.k_values == (1, 10, 100)


def test_asr_at_k_keys_match_k_values():
    p = _mixed_profile("p", 2, 2)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 4)
    assert set(result.asr_at_k.keys()) == set(bench.k_values)


def test_asr_at_1_equals_wins_over_attempts():
    p = _mixed_profile("p", 1, 3)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 4)
    assert result.total_attempts == 4
    assert result.wins == 1
    assert result.asr_at_k[1] == pytest.approx(0.25)


def test_asr_at_10_ge_asr_at_1():
    p = _mixed_profile("p", 1, 3)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 4)
    assert result.asr_at_k[10] >= result.asr_at_k[1]


def test_asr_at_k_monotone_non_decreasing():
    p = _mixed_profile("p", 1, 9)
    bench = ARTBenchmark(
        stub_judge_fn,
        stub_target_fn,
        attacker_profiles=(p,),
        k_values=(1, 2, 5, 10, 50, 100),
    )
    result = bench.run_attacker("p", 10)
    ks = sorted(result.asr_at_k.keys())
    vals = [result.asr_at_k[k] for k in ks]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


def test_run_all_returns_per_attacker():
    p1 = _all_unsafe_profile("a", 3)
    p2 = _all_safe_profile("b", 3)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p1, p2))
    results = bench.run_all(3)
    assert set(results.keys()) == {"a", "b"}
    assert isinstance(results["a"], ARTResult)
    assert results["a"].wins == 3
    assert results["b"].wins == 0


def test_empty_attempts_yields_zero_asr():
    p = _profile("p", [])
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 10)
    assert result.total_attempts == 0
    assert result.wins == 0
    assert all(v == 0.0 for v in result.asr_at_k.values())


def test_all_success_asr_approaches_one_as_k_grows():
    p = _all_unsafe_profile("p", 5)
    bench = ARTBenchmark(
        stub_judge_fn,
        stub_target_fn,
        attacker_profiles=(p,),
        k_values=(1, 10, 100, 1000),
    )
    result = bench.run_attacker("p", 5)
    # p_hat == 1.0 -> closed form clamps to 1.0 for any k >= 1.
    for k in (1, 10, 100, 1000):
        assert result.asr_at_k[k] == pytest.approx(1.0)


def test_custom_k_values_respected_and_sorted_unique():
    p = _mixed_profile("p", 2, 2)
    bench = ARTBenchmark(
        stub_judge_fn,
        stub_target_fn,
        attacker_profiles=(p,),
        k_values=(5, 2, 2, 20),
    )
    assert bench.k_values == (2, 5, 20)
    result = bench.run_attacker("p", 4)
    assert set(result.asr_at_k.keys()) == {2, 5, 20}


def test_judge_raising_is_recorded_as_failure():
    def bad_judge(prompt, response):
        raise RuntimeError("judge blew up")

    p = _all_unsafe_profile("p", 4)
    bench = ARTBenchmark(bad_judge, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 4)
    assert result.total_attempts == 4
    assert result.wins == 0
    assert result.asr_at_k[1] == pytest.approx(0.0)


def test_target_raising_is_recorded_as_failure():
    def bad_target(prompt):
        raise RuntimeError("target blew up")

    p = _all_unsafe_profile("p", 4)
    bench = ARTBenchmark(stub_judge_fn, bad_target, attacker_profiles=(p,))
    result = bench.run_attacker("p", 4)
    assert result.total_attempts == 4
    assert result.wins == 0


def test_unicode_prompt_survives_roundtrip():
    prompts = ["UNSAFE:日本語テスト", "safe-emoji-🌸"]
    p = _profile("p", prompts)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 2)
    assert result.total_attempts == 2
    assert result.wins == 1


def test_determinism_same_inputs_same_outputs():
    p = _mixed_profile("p", 2, 3)
    bench_a = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    bench_b = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    r1 = bench_a.run_attacker("p", 5)
    r2 = bench_b.run_attacker("p", 5)
    assert r1 == r2


def test_default_attacker_profiles_has_at_least_three_with_ten_attempts():
    assert len(DEFAULT_ATTACKER_PROFILES) >= 3
    for profile in DEFAULT_ATTACKER_PROFILES:
        assert isinstance(profile, AttackerProfile)
        assert len(profile.attempts) >= 10


def test_art_result_consistency_wins_le_total_attempts():
    p = _mixed_profile("p", 2, 3)
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn, attacker_profiles=(p,))
    result = bench.run_attacker("p", 5)
    assert result.wins <= result.total_attempts
    assert result.total_attempts == 5


def test_unknown_attacker_id_raises():
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn)
    with pytest.raises(KeyError):
        bench.run_attacker("no-such-id", 1)


def test_run_all_with_defaults_runs_cleanly():
    bench = ARTBenchmark(stub_judge_fn, stub_target_fn)
    results = bench.run_all(3)
    assert len(results) == len(DEFAULT_ATTACKER_PROFILES)
    for rid, res in results.items():
        assert isinstance(res, ARTResult)
        assert res.attacker_id == rid
        assert res.total_attempts == 3
