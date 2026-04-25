"""Tests for scenario sweeper."""
from __future__ import annotations

import pytest

from src.simulation.scenario_sweeper import ScenarioSweeper, SimConfig, ScenarioResult


class TestScenarioSweeper:
    def test_generate_configs(self):
        sweeper = ScenarioSweeper(runner=lambda c: {})
        configs = sweeper.generate_configs({"x": [1, 2], "y": [3, 4]})
        assert len(configs) == 4

    def test_run_all_succeed(self):
        def runner(c: SimConfig) -> dict[str, float]:
            return {"score": float(c.params.get("x", 0))}

        sweeper = ScenarioSweeper(runner=runner)
        configs = [SimConfig({"x": 1}), SimConfig({"x": 2})]
        results = sweeper.run(configs)
        assert len(results) == 2
        assert results[0].metrics["score"] == 1.0

    def test_run_catches_exceptions(self):
        def runner(c: SimConfig) -> dict[str, float]:
            raise ValueError("fail")
        sweeper = ScenarioSweeper(runner=runner)
        results = sweeper.run([SimConfig({"x": 1})])
        assert results[0].error is not None

    def test_best_returns_max(self):
        def runner(c: SimConfig) -> dict[str, float]:
            return {"score": float(c.params.get("v", 0))}
        sweeper = ScenarioSweeper(runner=runner)
        sweeper.run([SimConfig({"v": 1}), SimConfig({"v": 5}), SimConfig({"v": 3})])
        best = sweeper.best("score")
        assert best is not None
        assert best.config.params["v"] == 5

    def test_best_empty_returns_none(self):
        sweeper = ScenarioSweeper(runner=lambda c: {})
        assert sweeper.best("score") is None