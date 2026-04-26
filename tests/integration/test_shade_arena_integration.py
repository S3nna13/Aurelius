"""Integration test for SHADE-Arena harness registry + config wiring."""

from __future__ import annotations

from src.eval import (
    BENCHMARK_REGISTRY,
    DEFAULT_SHADE_SCENARIOS,
    METRIC_REGISTRY,
    MainTaskSpec,
    ShadeArenaHarness,
    ShadeArenaResult,
    SideTaskSpec,
    eval_shade_arena_enabled,
    stub_agent_fn,
    stub_monitor_fn,
)
from src.model.config import AureliusConfig


def test_shade_arena_registered():
    assert "shade_arena" in METRIC_REGISTRY
    assert "shade_arena" in BENCHMARK_REGISTRY
    assert METRIC_REGISTRY["shade_arena"] is ShadeArenaHarness
    assert BENCHMARK_REGISTRY["shade_arena"] is ShadeArenaHarness


def test_module_level_config_flag_defaults_off():
    assert eval_shade_arena_enabled is False


def test_model_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.eval_shade_arena_enabled is False


def test_default_catalog_runs_end_to_end():
    harness = ShadeArenaHarness(list(DEFAULT_SHADE_SCENARIOS))
    results = harness.run_all(stub_agent_fn, stub_monitor_fn)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, ShadeArenaResult)
        assert isinstance(r.scenario_id, str) and "::" in r.scenario_id
        assert 0.0 <= r.stealth_score <= 1.0


def test_default_catalog_shape_exposed_via_package():
    assert len(DEFAULT_SHADE_SCENARIOS) == 3
    for m, s in DEFAULT_SHADE_SCENARIOS:
        assert isinstance(m, MainTaskSpec)
        assert isinstance(s, SideTaskSpec)
