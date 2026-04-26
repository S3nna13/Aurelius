"""Integration test for the Crescendo multi-turn jailbreak probe."""

from __future__ import annotations

from src.eval import (
    BENCHMARK_REGISTRY,
    METRIC_REGISTRY,
    CrescendoProbe,
    CrescendoResult,
    eval_crescendo_probe_enabled,
)
from src.eval.crescendo_probe import stub_judge_fn


def _synthetic_target(history):
    turn = sum(1 for m in history if m.get("role") == "user")
    if turn < 2:
        return "General safe overview of the seed topic."
    return "Detailed on-topic walkthrough, step-by-step, addressing the objective explicitly."


def test_crescendo_registered_under_eval_registries():
    assert "crescendo_probe" in METRIC_REGISTRY
    assert "crescendo_probe" in BENCHMARK_REGISTRY
    assert METRIC_REGISTRY["crescendo_probe"] is CrescendoProbe
    assert BENCHMARK_REGISTRY["crescendo_probe"] is CrescendoProbe


def test_crescendo_config_flag_defaults_off():
    assert eval_crescendo_probe_enabled is False


def test_3_turn_probe_against_synthetic_target_and_judge():
    probe = CrescendoProbe(max_turns=3, max_backtracks=0)
    result = probe.run(
        "discuss a broad educational subject",
        _synthetic_target,
        stub_judge_fn,
        seed_topic="general science",
    )
    assert isinstance(result, CrescendoResult)
    assert len(result.turns) <= 3
    assert result.turns, "expected at least one turn"
    assert isinstance(result.succeeded, bool)
    assert isinstance(result.final_score, float)
    # Each turn exposes the documented shape.
    for t in result.turns:
        assert isinstance(t.attacker_message, str) and t.attacker_message
        assert isinstance(t.target_response, str) and t.target_response
        assert {"success", "score", "rationale"} == set(t.judge_verdict)
