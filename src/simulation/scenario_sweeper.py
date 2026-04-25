"""Parameter sweep runner for simulation experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SimConfig:
    """Single simulation configuration."""
    params: dict[str, Any]
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.tag:
            self.tag = "_".join(f"{k}={v}" for k, v in sorted(self.params.items()))


@dataclass
class ScenarioResult:
    """Result of running a single simulation scenario."""
    config: SimConfig
    metrics: dict[str, float]
    error: str | None = None


@dataclass
class ScenarioSweeper:
    """Run a parameter sweep over simulation configs."""

    runner: Callable[[SimConfig], dict[str, float]]
    _results: list[ScenarioResult] = field(default_factory=list, repr=False)

    def generate_configs(self, grid: dict[str, list[Any]]) -> list[SimConfig]:
        import itertools
        keys = list(grid.keys())
        values = list(grid.values())
        configs = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            configs.append(SimConfig(params=params))
        return configs

    def run(self, configs: list[SimConfig]) -> list[ScenarioResult]:
        self._results = []
        for config in configs:
            try:
                metrics = self.runner(config)
                self._results.append(ScenarioResult(config=config, metrics=metrics))
            except Exception as e:
                self._results.append(
                    ScenarioResult(config=config, metrics={}, error=str(e))
                )
        return self._results

    def best(self, metric: str, maximize: bool = True) -> ScenarioResult | None:
        if not self._results:
            return None
        key = lambda r: r.metrics.get(metric, float("-inf") if maximize else float("inf"))
        return max(self._results, key=key) if maximize else min(self._results, key=key)


SCENARIO_SWEEPER = None