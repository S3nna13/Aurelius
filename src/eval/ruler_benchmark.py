"""RULER long-context benchmark (arXiv:2404.06654).

Implements a lightweight subset of the RULER task suite from Hsieh et al.
(2024), "What's the Real Context Size of Your Long-Context Language
Models". RULER extends vanilla needle-in-a-haystack (see the sibling
`needle_in_haystack.py`) with distractor needles and five additional task
families that stress different long-context behaviors:

* Multi-key NIAH -- many (key, value) pairs buried in filler; query one.
* Multi-value NIAH -- one key with several values; retrieve all of them.
* Variable tracking -- chained variable assignments amid code-like noise;
  return the final value of a target variable.
* Common-words extraction -- the K words that appear most often in a
  populated word list, against many singleton distractors.
* Aggregation -- numbers embedded in prose; compute sum or max.

Each `build_*` method returns a `(prompt, expected)` pair. The `evaluate`
method accepts an injectable `generate_fn: Callable[[str], str]` so the
benchmark is unit-testable without a running model (tests pass an oracle
stub that reads the expected answer directly out of the prompt).

Design notes
------------
* Pure stdlib (`random`, `string`). No torch/transformers/datasets.
* All randomness is routed through a local `random.Random(seed)` so every
  builder is deterministic given its seed.
* `context_tokens` is approximated at 4 chars/token, matching
  `needle_in_haystack.py`. Exact tokenization is the caller's
  responsibility.
* Scoring is deliberately permissive (case-insensitive substring match on
  the expected answer) rather than strict equality -- RULER's own report
  uses a similar recall-style metric.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Sequence

# Shared lorem-like base sentences. Deterministic, self-contained.
_BASE_SENTENCES: tuple[str, ...] = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse.",
    "Excepteur sint occaecat cupidatat non proident sunt in culpa.",
    "Curabitur pretium tincidunt lacus nulla gravida orci a odio.",
    "Nullam varius turpis et commodo pharetra est eros bibendum elit.",
    "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices.",
    "Pellentesque habitant morbi tristique senectus et netus et malesuada.",
    "Aenean lacinia bibendum nulla sed consectetur pellentesque nibh.",
)

# Word pool for common-words extraction. All lowercase ascii, fixed.
_WORD_POOL: tuple[str, ...] = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
    "apple",
    "banana",
    "cherry",
    "date",
    "elder",
    "fig",
    "grape",
    "honey",
    "iris",
    "jade",
    "kiwi",
    "lotus",
    "mango",
    "nectar",
    "olive",
    "peach",
    "quince",
    "raven",
    "saffron",
    "thyme",
    "umber",
    "vanilla",
    "willow",
    "xenon",
)

APPROX_CHARS_PER_TOKEN = 4


# ----------------------------------------------------------------------
# Filler helpers
# ----------------------------------------------------------------------
def _filler(char_budget: int, rng: random.Random) -> str:
    """Deterministic lorem-like filler of ~char_budget chars."""
    if char_budget <= 0:
        return ""
    sentences = list(_BASE_SENTENCES)
    rng.shuffle(sentences)
    parts: list[str] = []
    total = 0
    i = 0
    while total < char_budget:
        s = sentences[i % len(sentences)]
        parts.append(s)
        total += len(s) + 1
        i += 1
    text = " ".join(parts)
    return text[:char_budget]


def _random_key(rng: random.Random, n_chars: int = 8) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=n_chars))


def _random_value(rng: random.Random, n_digits: int = 7) -> str:
    return "".join(rng.choices(string.digits, k=n_digits))


def _sprinkle(pieces: Sequence[str], filler: str, rng: random.Random) -> str:
    """Distribute `pieces` into `filler` at roughly even positions.

    Sentence-boundary aware: splits filler on ". " and interleaves the
    pieces between sentence fragments so nothing lands mid-word.
    """
    sents = [s for s in filler.split(". ") if s]
    n_pieces = len(pieces)
    if n_pieces == 0:
        return filler
    if not sents:
        return " ".join(list(pieces))
    # If we have more pieces than slots, allow repeated slots
    # (multiple pieces at the same boundary) so nothing is dropped.
    n_slots = len(sents) + 1
    if n_pieces <= n_slots:
        pool = list(range(n_slots))
        rng.shuffle(pool)
        chosen = sorted(pool[:n_pieces])
    else:
        # Distribute pieces across all slots roughly evenly.
        chosen = sorted(rng.choices(range(n_slots), k=n_pieces))
    # Rebuild, inserting pieces at chosen indices.
    out: list[str] = []
    pi = 0
    for i in range(len(sents) + 1):
        while pi < n_pieces and chosen[pi] == i:
            out.append(pieces[pi])
            pi += 1
        if i < len(sents):
            s = sents[i]
            if not s.endswith("."):
                s = s + "."
            out.append(s)
    return " ".join(out)


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
class RULERBenchmark:
    """RULER benchmark runner.

    Builder methods return ``(prompt, expected)``. Expected answers are
    strings, lists of strings, or numbers depending on the task.
    """

    TASKS: tuple[str, ...] = (
        "multi_key_niah",
        "multi_value_niah",
        "variable_tracking",
        "common_words_extraction",
        "aggregation_sum",
        "aggregation_max",
    )

    def __init__(self, approx_chars_per_token: int = APPROX_CHARS_PER_TOKEN) -> None:
        if approx_chars_per_token <= 0:
            raise ValueError(
                f"approx_chars_per_token must be positive, got {approx_chars_per_token!r}"
            )
        self.approx_chars_per_token = approx_chars_per_token

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _char_budget(self, context_tokens: int) -> int:
        if not isinstance(context_tokens, int) or context_tokens <= 0:
            raise ValueError(f"context_tokens must be a positive int, got {context_tokens!r}")
        return context_tokens * self.approx_chars_per_token

    def build_multi_key_niah(
        self, context_tokens: int, n_keys: int = 4, seed: int = 0
    ) -> tuple[str, str]:
        """Multi-key NIAH: insert n_keys (key, value) pairs; query one."""
        if not isinstance(n_keys, int) or n_keys <= 0:
            raise ValueError(f"n_keys must be a positive int, got {n_keys!r}")
        rng = random.Random(seed)  # noqa: S311
        budget = self._char_budget(context_tokens)

        # Generate distinct keys.
        keys: list[str] = []
        seen: set = set()
        while len(keys) < n_keys:
            k = _random_key(rng)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        values = [_random_value(rng) for _ in range(n_keys)]
        target_idx = rng.randrange(n_keys)
        target_key = keys[target_idx]
        target_value = values[target_idx]

        needle_sents = [f"The code for {k} is {v}." for k, v in zip(keys, values)]
        reserved = sum(len(s) + 1 for s in needle_sents) + 200
        filler_budget = max(1, budget - reserved)
        filler = _filler(filler_budget, rng)
        body = _sprinkle(needle_sents, filler, rng)

        prompt = body + f"\n\nQuestion: What is the code for {target_key}?\nAnswer:"
        return prompt, target_value

    def build_multi_value_niah(
        self, context_tokens: int, n_values: int = 3, seed: int = 0
    ) -> tuple[str, list[str]]:
        """Multi-value NIAH: one key, several values; retrieve them all."""
        if not isinstance(n_values, int) or n_values <= 0:
            raise ValueError(f"n_values must be a positive int, got {n_values!r}")
        rng = random.Random(seed)  # noqa: S311
        budget = self._char_budget(context_tokens)

        target_key = _random_key(rng)
        # Also emit a few distractor (other-key, value) pairs to make the
        # task nontrivial even at short contexts.
        distractor_keys = [_random_key(rng) for _ in range(max(2, n_values))]
        values = [_random_value(rng) for _ in range(n_values)]
        distractor_values = [_random_value(rng) for _ in range(len(distractor_keys))]

        needle_sents = [f"A value for {target_key} is {v}." for v in values]
        distractor_sents = [
            f"A value for {k} is {v}." for k, v in zip(distractor_keys, distractor_values)
        ]
        all_sents = needle_sents + distractor_sents
        rng.shuffle(all_sents)

        reserved = sum(len(s) + 1 for s in all_sents) + 200
        filler_budget = max(1, budget - reserved)
        filler = _filler(filler_budget, rng)
        body = _sprinkle(all_sents, filler, rng)

        prompt = body + f"\n\nQuestion: List all values for {target_key}.\nAnswer:"
        return prompt, list(values)

    def build_variable_tracking(
        self,
        context_tokens: int,
        n_vars: int = 5,
        chain_length: int = 3,
        seed: int = 0,
    ) -> tuple[str, str]:
        """Variable tracking: X1=const; X2=X1; ... return final value of target."""
        if not isinstance(n_vars, int) or n_vars <= 0:
            raise ValueError(f"n_vars must be a positive int, got {n_vars!r}")
        if not isinstance(chain_length, int) or chain_length <= 0:
            raise ValueError(f"chain_length must be a positive int, got {chain_length!r}")
        rng = random.Random(seed)  # noqa: S311
        budget = self._char_budget(context_tokens)

        # Build n_vars independent chains; track the first chain as target.
        assignments: list[str] = []
        target_tail_name: str | None = None
        target_value: str | None = None
        for vi in range(n_vars):
            root_val = _random_value(rng)
            names = [f"var_{vi}_{step}" for step in range(chain_length)]
            # Seed assignment.
            assignments.append(f"{names[0]} = {root_val}")
            for step in range(1, chain_length):
                assignments.append(f"{names[step]} = {names[step - 1]}")
            if vi == 0:
                target_tail_name = names[-1]
                target_value = root_val

        assert target_tail_name is not None and target_value is not None  # noqa: S101

        rng.shuffle(assignments)
        code_sents = [a + "." for a in assignments]

        reserved = sum(len(s) + 1 for s in code_sents) + 200
        filler_budget = max(1, budget - reserved)
        filler = _filler(filler_budget, rng)
        body = _sprinkle(code_sents, filler, rng)

        prompt = body + f"\n\nQuestion: What is the final value of {target_tail_name}?\nAnswer:"
        return prompt, target_value

    def build_common_words_extraction(
        self, context_tokens: int, n_common: int = 5, seed: int = 0
    ) -> tuple[str, list[str]]:
        """Common-words extraction: K words repeated often amid singletons."""
        if not isinstance(n_common, int) or n_common <= 0:
            raise ValueError(f"n_common must be a positive int, got {n_common!r}")
        if n_common > len(_WORD_POOL) // 2:
            raise ValueError(f"n_common={n_common} too large for pool of {len(_WORD_POOL)}")
        rng = random.Random(seed)  # noqa: S311
        budget = self._char_budget(context_tokens)

        pool = list(_WORD_POOL)
        rng.shuffle(pool)
        common = pool[:n_common]
        singletons = pool[n_common:]

        # Emit each common word many times, each singleton once.
        common_count = 10
        words: list[str] = []
        for w in common:
            words.extend([w] * common_count)
        words.extend(singletons)
        rng.shuffle(words)

        word_block = " ".join(words)
        # Scale filler so total fits the budget.
        reserved = len(word_block) + 300
        filler_budget = max(1, budget - reserved)
        filler = _filler(filler_budget, rng)
        body = filler + " " + word_block

        prompt = body + f"\n\nQuestion: List the {n_common} most frequent words above.\nAnswer:"
        return prompt, list(common)

    def build_aggregation(
        self, context_tokens: int, operation: str = "sum", seed: int = 0
    ) -> tuple[str, int]:
        """Aggregation: embed numbers; compute sum or max."""
        if operation not in ("sum", "max"):
            raise ValueError(f"operation must be 'sum' or 'max', got {operation!r}")
        rng = random.Random(seed)  # noqa: S311
        budget = self._char_budget(context_tokens)

        n_numbers = 12
        numbers = [rng.randint(1, 999) for _ in range(n_numbers)]
        if operation == "sum":
            expected = sum(numbers)
        else:
            expected = max(numbers)

        num_sents = [f"Observation: the recorded value is {n}." for n in numbers]
        reserved = sum(len(s) + 1 for s in num_sents) + 200
        filler_budget = max(1, budget - reserved)
        filler = _filler(filler_budget, rng)
        body = _sprinkle(num_sents, filler, rng)

        op_word = "sum" if operation == "sum" else "maximum"
        prompt = (
            body + f"\n\nQuestion: Compute the {op_word} of all recorded values above.\nAnswer:"
        )
        return prompt, expected

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _build_task(
        self, task: str, context_tokens: int, seed: int
    ) -> tuple[str, str | list[str] | int]:
        if task == "multi_key_niah":
            return self.build_multi_key_niah(context_tokens, seed=seed)
        if task == "multi_value_niah":
            return self.build_multi_value_niah(context_tokens, seed=seed)
        if task == "variable_tracking":
            return self.build_variable_tracking(context_tokens, seed=seed)
        if task == "common_words_extraction":
            return self.build_common_words_extraction(context_tokens, seed=seed)
        if task == "aggregation_sum":
            return self.build_aggregation(context_tokens, operation="sum", seed=seed)
        if task == "aggregation_max":
            return self.build_aggregation(context_tokens, operation="max", seed=seed)
        raise ValueError(f"unknown task {task!r}; known: {self.TASKS}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _check(expected: str | list[str] | int, response: str) -> bool:
        if not isinstance(response, str):
            raise TypeError(f"generate_fn must return str, got {type(response).__name__}")
        r = response.lower()
        if isinstance(expected, list):
            return all(str(v).lower() in r for v in expected)
        return str(expected).lower() in r

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        tasks: list[str],
        context_lengths: list[int],
        samples_per: int = 3,
    ) -> dict[str, dict[str, object]]:
        """Run a task x context_length grid, `samples_per` seeds each.

        Returns a dict keyed by task with per-length pass rates and the raw
        cells.
        """
        if not callable(generate_fn):
            raise ValueError("generate_fn must be callable")
        if not tasks:
            raise ValueError("tasks must be a non-empty list")
        if not context_lengths:
            raise ValueError("context_lengths must be a non-empty list")
        if not isinstance(samples_per, int) or samples_per <= 0:
            raise ValueError(f"samples_per must be a positive int, got {samples_per!r}")
        for task in tasks:
            if task not in self.TASKS:
                raise ValueError(f"unknown task {task!r}; known: {self.TASKS}")
        for L in context_lengths:
            if not isinstance(L, int) or L <= 0:
                raise ValueError(f"context length must be positive int, got {L!r}")

        results: dict[str, dict[str, object]] = {}
        for task in tasks:
            cells: list[dict[str, object]] = []
            for L in context_lengths:
                for s in range(samples_per):
                    prompt, expected = self._build_task(task, L, seed=s)
                    response = generate_fn(prompt)
                    passed = self._check(expected, response)
                    cells.append(
                        {
                            "task": task,
                            "context_tokens": L,
                            "seed": s,
                            "expected": expected,
                            "response": response,
                            "pass": passed,
                        }
                    )
            n = len(cells)
            passes = sum(1 for c in cells if c["pass"])
            results[task] = {
                "pass_rate": passes / n if n else 0.0,
                "n": n,
                "cells": cells,
            }
        return results

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def score_per_task(results: dict[str, dict[str, object]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for task, info in results.items():
            rate = info.get("pass_rate", 0.0)
            if not isinstance(rate, (int, float)):
                raise TypeError(
                    f"pass_rate for {task!r} must be numeric, got {type(rate).__name__}"
                )
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"pass_rate for {task!r} out of range: {rate!r}")
            out[task] = float(rate)
        return out

    @classmethod
    def overall_score(cls, results: dict[str, dict[str, object]]) -> float:
        per = cls.score_per_task(results)
        if not per:
            return 0.0
        return sum(per.values()) / len(per)


# Silence linter: `string` is re-exported for composability by callers.
_ = string
