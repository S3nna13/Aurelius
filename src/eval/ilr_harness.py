"""Instance-Level Randomization (ILR) harness for Aurelius evaluation.

ILR (EMNLP 2025) randomizes prompt format, option labels, and few-shot
examples per evaluation instance, runs multiple trials, and aggregates.
This reduces variance and unfair rankings with <50% compute vs multiple
full runs.

The harness wraps any existing benchmark object (anything with an
``.evaluate()`` method) and works without modifying the wrapped module.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ILRConfig:
    """Configuration for Instance-Level Randomization."""

    n_trials: int = 3
    seed: int = 42
    randomize_prompt_format: bool = True
    shuffle_option_labels: bool = True
    randomize_fewshot_order: bool = True


class ILRHarness:
    """Wrap a benchmark and evaluate with per-instance randomization.

    Args:
        benchmark: Any object with an ``evaluate(data)`` method.
        config: ``ILRConfig`` controlling randomization and trials.
    """

    def __init__(self, benchmark: Any, config: ILRConfig | None = None) -> None:
        self.benchmark = benchmark
        self.config = config or ILRConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, data: Any) -> dict[str, Any]:
        """Run ``n_trials`` randomized evaluations and aggregate results.

        Args:
            data: The input passed to the wrapped benchmark's ``evaluate``.

        Returns:
            Aggregated dict with mean metrics, per-key std-dev confidence
            intervals, and an ``ilr_details`` key containing raw trial data.
        """
        trials: list[dict[str, Any]] = []
        for trial_idx in range(self.config.n_trials):
            randomized = self._randomize(
                copy.deepcopy(data) if not hasattr(data, "__dict__") else data, trial_idx
            )
            raw = self.benchmark.evaluate(randomized)
            trials.append(self._to_dict(raw))
        return self._aggregate(trials)

    # ------------------------------------------------------------------
    # Randomization helpers
    # ------------------------------------------------------------------

    def _randomize(self, obj: Any, trial_idx: int) -> Any:
        """Apply all enabled randomizations to *obj* for a single trial."""
        trial_seed = self.config.seed + trial_idx
        rng = random.Random(trial_seed)
        return self._randomize_object(obj, rng)

    def _randomize_object(self, obj: Any, rng: random.Random) -> Any:
        """Recursively traverse and randomize *obj*."""
        if isinstance(obj, dict):
            obj = self._maybe_shuffle_mcq(obj, rng)
            obj = self._maybe_shuffle_fewshot_dict(obj, rng)
            return {k: self._randomize_object(v, rng) for k, v in obj.items()}

        if isinstance(obj, list):
            obj = self._maybe_shuffle_example_list(obj, rng)
            return [self._randomize_object(item, rng) for item in obj]

        if isinstance(obj, str) and self.config.randomize_prompt_format:
            return self._randomize_prompt(obj, rng)

        # Pass through untouched for tensors, ints, floats, etc.
        return obj

    # -- MCQ option shuffling ------------------------------------------

    def _is_mcq_dict(self, d: dict) -> bool:
        return (
            "choices" in d
            and isinstance(d["choices"], list)
            and len(d["choices"]) > 1
            and ("correct_idx" in d or "answer_idx" in d)
        )

    def _maybe_shuffle_mcq(self, d: dict, rng: random.Random) -> dict:
        if not self.config.shuffle_option_labels or not self._is_mcq_dict(d):
            return d
        d = dict(d)
        choices = list(d["choices"])
        n = len(choices)
        perm = list(range(n))
        rng.shuffle(perm)
        d["choices"] = [choices[i] for i in perm]
        # Build inverse permutation to remap correct / answer indices
        inv = [0] * n
        for new_pos, old_pos in enumerate(perm):
            inv[old_pos] = new_pos
        if "correct_idx" in d:
            d["correct_idx"] = inv[int(d["correct_idx"])]
        if "answer_idx" in d:
            d["answer_idx"] = inv[int(d["answer_idx"])]
        return d

    # -- Few-shot order randomization ----------------------------------

    def _is_example_list(self, lst: list) -> bool:
        if not lst or not all(isinstance(x, dict) for x in lst):
            return False
        first = lst[0]
        return (
            "question" in first
            and "choices" in first
            and ("answer_idx" in first or "correct_idx" in first)
        )

    def _maybe_shuffle_example_list(self, lst: list, rng: random.Random) -> list:
        if not self.config.randomize_fewshot_order or not self._is_example_list(lst):
            return lst
        lst = list(lst)
        rng.shuffle(lst)
        return lst

    def _maybe_shuffle_fewshot_dict(self, d: dict, rng: random.Random) -> dict:
        """Shuffle values keyed by common few-shot pool names."""
        if not self.config.randomize_fewshot_order:
            return d
        for key in ("few_shot_pool", "examples", "shots", "demonstrations"):
            if key in d and isinstance(d[key], list):
                d = dict(d)
                pool = list(d[key])
                if pool and all(isinstance(x, dict) for x in pool):
                    rng.shuffle(pool)
                    d[key] = pool
                break
        return d

    # -- Prompt format randomization -----------------------------------

    def _randomize_prompt(self, text: str, rng: random.Random) -> str:
        """Apply 1-2 lightweight formatting randomizations to *text*."""
        strategies = [
            self._strategy_whitespace,
            self._strategy_delimiter,
            self._strategy_punctuation_spaces,
        ]
        n_strategies = rng.choice([1, 2])
        for strat in rng.sample(strategies, n_strategies):
            text = strat(text, rng)
        return text

    @staticmethod
    def _strategy_whitespace(text: str, rng: random.Random) -> str:
        if rng.random() < 0.5:
            return text.strip()
        return "\n" + text + "\n"

    @staticmethod
    def _strategy_delimiter(text: str, rng: random.Random) -> str:
        delimiters = ["\n\n", "\n", " ", " | ", " --- "]
        replacement = rng.choice(delimiters)
        return re.sub(r"\n\s*\n", replacement, text)

    @staticmethod
    def _strategy_punctuation_spaces(text: str, rng: random.Random) -> str:
        for p in (",", ".", ":", ";", "?"):
            r = rng.random()
            if r < 0.15:
                text = text.replace(p, p + " ")
            elif r < 0.30:
                text = text.replace(p, " " + p)
        return text

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "__dict__"):
            return raw.__dict__
        return {"value": raw}

    def _aggregate(self, trials: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate trial results.

        * Continuous metrics -> mean with std-dev CI.
        * Discrete metrics (e.g. accuracy) -> mean with std-dev CI.
          True per-instance majority vote requires per-instance predictions
          that generic wrappers cannot always access, so we report the mean
          accuracy across trials and its std deviation as a confidence proxy.
        """
        if not trials:
            return {}

        all_keys = set().union(*(t.keys() for t in trials))
        aggregated: dict[str, Any] = {}
        numeric_keys: list[str] = []

        for key in all_keys:
            values = []
            for t in trials:
                val = t.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    values.append(float(val))
            if values:
                numeric_keys.append(key)
                arr = np.array(values)
                aggregated[key] = float(arr.mean())
                if len(arr) > 1:
                    aggregated[f"{key}_std"] = float(arr.std(ddof=1))

        # Preserve non-numeric keys that are stable across trials
        for key in all_keys:
            if key in aggregated:
                continue
            vals = [t.get(key) for t in trials]
            if all(v == vals[0] for v in vals):
                aggregated[key] = vals[0]

        aggregated["ilr_details"] = {
            "n_trials": len(trials),
            "trials": trials,
            "config": {
                "n_trials": self.config.n_trials,
                "seed": self.config.seed,
                "randomize_prompt_format": self.config.randomize_prompt_format,
                "shuffle_option_labels": self.config.shuffle_option_labels,
                "randomize_fewshot_order": self.config.randomize_fewshot_order,
            },
        }
        return aggregated
