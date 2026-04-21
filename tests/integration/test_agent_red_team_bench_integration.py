"""Integration test for the Agent Red Teaming (ART) benchmark."""
from __future__ import annotations

from src.eval import (
    METRIC_REGISTRY,
    BENCHMARK_REGISTRY,
    ARTBenchmark,
    ARTResult,
    AttackAttempt,
    AttackerProfile,
    ART_DEFAULT_ATTACKER_PROFILES,
    art_stub_target_fn,
    art_stub_judge_fn,
    eval_agent_red_team_bench_enabled,
)
from src.model.config import AureliusConfig


def test_agent_red_team_bench_registered():
    assert "agent_red_team_bench" in METRIC_REGISTRY
    assert "agent_red_team_bench" in BENCHMARK_REGISTRY
    assert METRIC_REGISTRY["agent_red_team_bench"] is ARTBenchmark
    assert BENCHMARK_REGISTRY["agent_red_team_bench"] is ARTBenchmark


def test_agent_red_team_bench_config_flag_defaults_off():
    assert eval_agent_red_team_bench_enabled is False
    cfg = AureliusConfig()
    assert cfg.eval_agent_red_team_bench_enabled is False


def test_default_attacker_profiles_exported_and_sane():
    assert len(ART_DEFAULT_ATTACKER_PROFILES) >= 3
    for profile in ART_DEFAULT_ATTACKER_PROFILES:
        assert isinstance(profile, AttackerProfile)
        assert len(profile.attempts) >= 10


def test_end_to_end_art_run_against_stubs():
    prompts = ["UNSAFE:a", "safe-x", "UNSAFE:b", "safe-y"]
    attempts = tuple(
        AttackAttempt(p, "", False, 0.0, "") for p in prompts
    )
    profile = AttackerProfile(
        id="mix", name="Mix", style="mixed", attempts=attempts
    )
    bench = ARTBenchmark(
        art_stub_judge_fn,
        art_stub_target_fn,
        attacker_profiles=(profile,),
        k_values=(1, 10, 100),
    )
    result = bench.run_attacker("mix", 4)
    assert isinstance(result, ARTResult)
    assert result.total_attempts == 4
    assert result.wins == 2
    assert result.asr_at_k[1] == 0.5
    assert result.asr_at_k[10] > result.asr_at_k[1]
