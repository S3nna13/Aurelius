"""Neural Architecture Search via Hyperband (Successive Halving).

Efficient hyperparameter search using random sampling + successive halving.
Each trial trains for increasing budgets; bad configurations are eliminated early.

Reference: Li et al. 2018 "Hyperband: A Novel Bandit-Based Approach to HPO"
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space."""

    # Each entry: param_name -> (low, high, log_scale)
    # log_scale=True: sample uniformly in log space (good for lr)
    continuous: dict[str, tuple[float, float, bool]] = field(default_factory=dict)

    # Discrete choices: param_name -> [option1, option2, ...]
    discrete: dict[str, list[Any]] = field(default_factory=dict)

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Sample one configuration from the search space."""
        rng = rng or random
        config = {}
        for name, (low, high, log_scale) in self.continuous.items():
            if log_scale:
                val = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                val = rng.uniform(low, high)
            config[name] = val
        for name, choices in self.discrete.items():
            config[name] = rng.choice(choices)
        return config


@dataclass
class Trial:
    """One NAS trial (one configuration)."""
    config: dict[str, Any]
    trial_id: int
    best_loss: float = float("inf")
    steps_trained: int = 0
    eliminated: bool = False


@dataclass
class NASConfig:
    n_initial_configs: int = 16    # initial number of random configs
    eta: int = 3                   # reduction factor (keep 1/eta each round)
    min_budget: int = 10           # initial training steps per config
    max_budget: int = 100          # max training steps (when winner is found)
    seed: int = 42


@dataclass
class NASResult:
    """Best configuration found by NAS."""
    best_config: dict[str, Any]
    best_loss: float
    n_trials: int
    total_steps: int
    trial_history: list[Trial]


class HyperbandSearcher:
    """Hyperband: successive halving with multiple brackets.

    Each bracket starts with a different (n, budget) tradeoff.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        eval_fn: Callable[[dict, int], float],  # (config, n_steps) -> validation_loss
        cfg: NASConfig | None = None,
    ) -> None:
        self.search_space = search_space
        self.eval_fn = eval_fn
        self.cfg = cfg or NASConfig()
        self.rng = random.Random(self.cfg.seed)
        self._trials: list[Trial] = []
        self._total_steps = 0

    def _successive_halving(
        self,
        configs: list[dict],
        budget: int,
        n_rungs: int,
    ) -> list[Trial]:
        """Run successive halving on a list of configs.

        Returns list of Trial objects (including eliminated ones).
        """
        cfg = self.cfg
        all_trials: list[Trial] = []

        # Create Trial objects for each config
        active_trials: list[Trial] = []
        for config in configs:
            trial_id = len(self._trials) + len(all_trials) + len(active_trials)
            trial = Trial(config=config, trial_id=trial_id)
            active_trials.append(trial)
            all_trials.append(trial)

        current_budget = budget

        for rung in range(n_rungs):
            # Evaluate each active trial at current_budget steps
            for trial in active_trials:
                additional_steps = current_budget - trial.steps_trained
                if additional_steps > 0:
                    loss = self.eval_fn(trial.config, current_budget)
                    trial.best_loss = loss
                    trial.steps_trained = current_budget
                    self._total_steps += additional_steps

            # Sort by loss, keep top 1/eta
            active_trials.sort(key=lambda t: t.best_loss)
            n_keep = max(1, len(active_trials) // cfg.eta)

            # Eliminate the bottom configs (unless last rung)
            if rung < n_rungs - 1:
                for trial in active_trials[n_keep:]:
                    trial.eliminated = True
                active_trials = active_trials[:n_keep]
                current_budget = min(current_budget * cfg.eta, cfg.max_budget)

        return all_trials

    def search(self) -> NASResult:
        """Run full Hyperband search.

        Number of brackets = floor(log_{eta}(max_budget/min_budget)) + 1
        For each bracket: sample n_i configs, run SHA with budget_i
        """
        cfg = self.cfg
        eta = cfg.eta

        # Number of brackets (s_max + 1)
        s_max = math.floor(math.log(cfg.max_budget / cfg.min_budget, eta))
        n_brackets = s_max + 1

        all_trials: list[Trial] = []

        for s in range(n_brackets - 1, -1, -1):
            # Number of configs and initial budget for this bracket
            # n_i = ceil(n_initial * eta^s / (s+1))
            n_i = math.ceil(cfg.n_initial_configs * (eta ** s) / (s + 1))
            # b_i = min_budget * eta^(s_max - s) — start lower for brackets with more configs
            b_i = cfg.min_budget * (eta ** (s_max - s))
            b_i = min(b_i, cfg.max_budget)

            # Number of rungs for this bracket
            n_rungs = s + 1

            # Sample configs for this bracket
            configs = [self.search_space.sample(self.rng) for _ in range(n_i)]

            # Run successive halving
            bracket_trials = self._successive_halving(configs, b_i, n_rungs)

            # Track trial IDs properly
            for trial in bracket_trials:
                trial.trial_id = len(all_trials)
                all_trials.append(trial)

        self._trials = all_trials

        # Find best trial
        best_trial = min(all_trials, key=lambda t: t.best_loss)

        return NASResult(
            best_config=best_trial.config,
            best_loss=best_trial.best_loss,
            n_trials=len(all_trials),
            total_steps=self._total_steps,
            trial_history=all_trials,
        )


def random_search(
    search_space: SearchSpace,
    eval_fn: Callable[[dict, int], float],
    n_trials: int,
    budget_per_trial: int,
    seed: int = 0,
) -> NASResult:
    """Simple random search baseline.

    Sample n_trials configs, evaluate each for budget_per_trial steps.
    Return best.
    """
    rng = random.Random(seed)
    trials: list[Trial] = []
    total_steps = 0

    for i in range(n_trials):
        config = search_space.sample(rng)
        loss = eval_fn(config, budget_per_trial)
        trial = Trial(
            config=config,
            trial_id=i,
            best_loss=loss,
            steps_trained=budget_per_trial,
            eliminated=False,
        )
        trials.append(trial)
        total_steps += budget_per_trial

    best_trial = min(trials, key=lambda t: t.best_loss)

    return NASResult(
        best_config=best_trial.config,
        best_loss=best_trial.best_loss,
        n_trials=n_trials,
        total_steps=total_steps,
        trial_history=trials,
    )
