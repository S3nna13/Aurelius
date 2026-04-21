"""Integration test for the many-shot jailbreak probe."""
from __future__ import annotations

from src.eval import (
    METRIC_REGISTRY,
    BENCHMARK_REGISTRY,
    ManyShotJailbreakProbe,
    MSJSweepResult,
    MSJ_DEFAULT_SHOT_LIBRARY,
    eval_many_shot_jailbreak_probe_enabled,
)
from src.eval.many_shot_jailbreak_probe import (
    stub_target_fn,
    stub_judge_fn,
)
from src.model.config import AureliusConfig


def test_many_shot_jailbreak_probe_registered():
    assert "many_shot_jailbreak_probe" in METRIC_REGISTRY
    assert "many_shot_jailbreak_probe" in BENCHMARK_REGISTRY
    assert (
        METRIC_REGISTRY["many_shot_jailbreak_probe"]
        is ManyShotJailbreakProbe
    )
    assert (
        BENCHMARK_REGISTRY["many_shot_jailbreak_probe"]
        is ManyShotJailbreakProbe
    )


def test_many_shot_jailbreak_probe_config_flag_defaults_off():
    assert eval_many_shot_jailbreak_probe_enabled is False
    cfg = AureliusConfig()
    assert cfg.eval_many_shot_jailbreak_probe_enabled is False


def test_default_shot_library_exported_and_sane():
    assert len(MSJ_DEFAULT_SHOT_LIBRARY) >= 20


def test_end_to_end_sweep_against_stubs():
    probe = ManyShotJailbreakProbe(
        max_shots=32, shot_schedule=(1, 4, 16, 32)
    )
    result = probe.run("integration objective", stub_target_fn, stub_judge_fn)
    assert isinstance(result, MSJSweepResult)
    assert result.first_success_shot == 16
    assert len(result.per_shot_results) == 4
