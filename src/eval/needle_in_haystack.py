"""Needle-in-a-Haystack (NIAH) long-context benchmark.

Implementation of Greg Kamradt's NIAH protocol with extensions inspired by
RULER (arXiv:2404.06654). The benchmark builds a long filler ("haystack"),
inserts a unique factual sentence (the "needle") at a controllable fractional
depth, and asks a question whose answer is the needle. Pass/fail is measured
by case-insensitive substring match of an answer key in the model's response.

Design notes
------------
* Pure-Python stdlib only (`random`, `string`). No torch / transformers /
  datasets dependency. Callers inject a `generate_fn: Callable[[str], str]`,
  so this module is trivially unit-testable without a running model.
* The default haystack filler is a deterministic repeating Lorem-like
  sentence. It accepts an optional `seed` that re-shuffles the filler
  sentences so multiple runs can be randomized while remaining reproducible.
* `context_tokens` is translated to characters via `approx_chars_per_token`
  (default 4 -- the conventional BPE heuristic). This keeps the module free
  of any tokenizer dependency; callers who care about exact token lengths
  should pre-compute their own character budget.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable

# Lorem-like base sentences. Deterministic, self-contained, no external IO.
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


def default_haystack_filler(char_budget: int, seed: int | None = None) -> str:
    """Build a deterministic filler of approximately `char_budget` characters.

    If `seed` is provided the ordering of base sentences is shuffled with a
    local `random.Random(seed)`; output is fully reproducible.
    """
    if char_budget <= 0:
        return ""
    sentences = list(_BASE_SENTENCES)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(sentences)
    # Repeat sentences until the budget is filled, joined by single spaces.
    parts: list[str] = []
    total = 0
    i = 0
    while total < char_budget:
        s = sentences[i % len(sentences)]
        parts.append(s)
        total += len(s) + 1  # +1 for the space separator
        i += 1
    filler = " ".join(parts)
    return filler[:char_budget]


class NeedleInHaystackBenchmark:
    """Needle-in-a-Haystack benchmark runner.

    Parameters
    ----------
    haystack_filler : Callable[[int], str] | None
        Function `f(char_budget) -> str` producing deterministic filler of
        approximately `char_budget` characters. If None, the module-level
        `default_haystack_filler` is used (unseeded).
    needle : str
        The unique fact to embed in the haystack.
    question : str
        The question appended to the prompt.
    answer_key : str
        Substring whose case-insensitive presence in the response marks a
        pass.
    """

    def __init__(
        self,
        haystack_filler: Callable[[int], str] | None = None,
        needle: str = "The magic number is 42.",
        question: str = "What is the magic number?",
        answer_key: str = "42",
    ) -> None:
        if not needle:
            raise ValueError("needle must be a non-empty string")
        if not question:
            raise ValueError("question must be a non-empty string")
        if not answer_key:
            raise ValueError("answer_key must be a non-empty string")
        self.haystack_filler = haystack_filler or default_haystack_filler
        self.needle = needle
        self.question = question
        self.answer_key = answer_key

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        context_tokens: int,
        depth: float,
        approx_chars_per_token: int = 4,
    ) -> str:
        """Build a prompt of ~`context_tokens` tokens with the needle at `depth`.

        `depth` is a fraction in [0,1]: 0.0 inserts the needle near the start
        of the haystack, 1.0 near the end. The needle is inserted on a line
        boundary -- i.e. between two filler sentences -- so sentence
        structure is preserved.
        """
        if not isinstance(context_tokens, int) or context_tokens <= 0:
            raise ValueError(f"context_tokens must be a positive int, got {context_tokens!r}")
        if not (0.0 <= depth <= 1.0):
            raise ValueError(f"depth must be in [0,1], got {depth!r}")
        if approx_chars_per_token <= 0:
            raise ValueError(
                f"approx_chars_per_token must be positive, got {approx_chars_per_token!r}"
            )

        # Reserve a portion of the budget for the question + needle; the rest
        # is filler. We do not aggressively trim -- final prompts are allowed
        # to slightly exceed `context_tokens * approx_chars_per_token` because
        # the needle and question are load-bearing and must not be elided.
        char_budget = context_tokens * approx_chars_per_token
        tail = "\n\nQuestion: " + self.question + "\nAnswer:"
        reserved = len(self.needle) + len(tail) + 2
        filler_budget = max(1, char_budget - reserved)

        filler = self.haystack_filler(filler_budget)
        # Split on sentence boundaries so the needle lands cleanly between
        # sentences rather than mid-word.
        sentences = [s for s in filler.split(". ") if s]
        if not sentences:
            sentences = [filler] if filler else [""]

        # Fractional index: depth=0 -> insert before first sentence,
        # depth=1 -> after last.
        n = len(sentences)
        idx = int(round(depth * n))
        idx = max(0, min(n, idx))

        before = sentences[:idx]
        after = sentences[idx:]
        rebuilt_before = ". ".join(before)
        rebuilt_after = ". ".join(after)
        pieces: list[str] = []
        if rebuilt_before:
            pieces.append(rebuilt_before.rstrip(". ") + ".")
        pieces.append(self.needle)
        if rebuilt_after:
            pieces.append(rebuilt_after.rstrip(". ") + ".")
        body = " ".join(pieces)
        return body + tail

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        context_lengths: list[int],
        depths: list[float],
    ) -> dict[tuple[int, float], dict[str, object]]:
        """Run the 2D grid (L, d) and return per-cell results.

        Returns a dict keyed by (L, d) with {"pass": bool, "response": str}.
        """
        if not callable(generate_fn):
            raise ValueError("generate_fn must be callable")
        if not context_lengths:
            raise ValueError("context_lengths must be a non-empty list")
        if not depths:
            raise ValueError("depths must be a non-empty list")
        for L in context_lengths:
            if not isinstance(L, int) or L <= 0:
                raise ValueError(f"context length must be positive int, got {L!r}")
        for d in depths:
            if not (0.0 <= d <= 1.0):
                raise ValueError(f"depth must be in [0,1], got {d!r}")

        key_lc = self.answer_key.lower()
        results: dict[tuple[int, float], dict[str, object]] = {}
        for L in context_lengths:
            for d in depths:
                prompt = self.build_prompt(L, d)
                response = generate_fn(prompt)
                if not isinstance(response, str):
                    raise TypeError(f"generate_fn must return str, got {type(response).__name__}")
                passed = key_lc in response.lower()
                results[(L, d)] = {"pass": passed, "response": response}
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def score(results: dict[tuple[int, float], dict[str, object]]) -> float:
        """Overall pass rate in [0,1]."""
        if not results:
            return 0.0
        passes = sum(1 for cell in results.values() if cell.get("pass"))
        return passes / len(results)

    @staticmethod
    def grid_report(results: dict[tuple[int, float], dict[str, object]]) -> str:
        """ASCII heatmap: rows = context length, cols = depth."""
        if not results:
            return "(empty grid)"
        lengths = sorted({L for (L, _) in results.keys()})
        depths = sorted({d for (_, d) in results.keys()})
        header = "L \\ d  | " + " ".join(f"{d:4.2f}" for d in depths)
        lines = [header, "-" * len(header)]
        for L in lengths:
            row_cells = []
            for d in depths:
                cell = results.get((L, d))
                if cell is None:
                    row_cells.append(" ?  ")
                else:
                    row_cells.append(" OK " if cell.get("pass") else " .. ")
            lines.append(f"{L:6d} | " + " ".join(row_cells))
        return "\n".join(lines)


# Silence linter for unused `string` import (kept available for filler
# extensions callers may compose).
_ = string
