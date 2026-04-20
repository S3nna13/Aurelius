"""Integration: synthetic jailbreak generator in eval registry."""

from __future__ import annotations

import src.eval as ev


def test_benchmark_registry_entry():
    assert ev.BENCHMARK_REGISTRY.get("synthetic_jailbreak") is ev.SyntheticJailbreakGenerator
    assert ev.METRIC_REGISTRY.get("synthetic_jailbreak") is ev.SyntheticJailbreakGenerator


def test_config_flag_off():
    from src.model.config import AureliusConfig

    assert AureliusConfig().eval_synthetic_jailbreak_generator_enabled is False
    assert ev.eval_synthetic_jailbreak_generator_enabled is False


def test_prior_eval_entries_exist():
    assert "humaneval" in ev.BENCHMARK_REGISTRY or "niah" in ev.BENCHMARK_REGISTRY


def test_smoke_generate():
    g = ev.SyntheticJailbreakGenerator()
    probes = g.generate(5, seed=1)
    assert len(probes) == 5
    assert all(len(p.text) > 0 for p in probes)
