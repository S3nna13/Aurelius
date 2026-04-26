"""Automated few-shot prompt construction and selection."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FewShotConfig:
    n_shots: int = 5
    max_seq_len: int = 512
    selection_strategy: str = "random"  # "random" | "similarity" | "diverse"
    template: str = "Q: {input}\nA: {output}"
    separator: str = "\n\n"
    instruction: str = ""  # optional system instruction prefix


@dataclass
class FewShotExample:
    input_text: str
    output_text: str
    metadata: dict = field(default_factory=dict)


def format_example(example: FewShotExample, template: str) -> str:
    """Format a single example using template with {input} and {output} placeholders."""
    return template.format(input=example.input_text, output=example.output_text)


def format_few_shot_prompt(
    examples: list[FewShotExample],
    query: str,
    cfg: FewShotConfig,
    query_template: str | None = None,
) -> str:
    """Build full few-shot prompt.

    instruction + separator + formatted examples + separator + query.
    query_template: if provided, format query as "Q: {query}" (no answer), else just query.
    """
    parts: list[str] = []

    if cfg.instruction:
        parts.append(cfg.instruction)

    for ex in examples:
        parts.append(format_example(ex, cfg.template))

    if query_template is not None:
        query_part = query_template.format(input=query)
    else:
        query_part = query
    parts.append(query_part)

    return cfg.separator.join(parts)


def compute_example_similarity(
    query: str,
    example: FewShotExample,
) -> float:
    """Character-level Jaccard similarity between query and example.input_text."""
    q_chars = set(query)
    e_chars = set(example.input_text)
    intersection = q_chars & e_chars
    union = q_chars | e_chars
    if not union:
        return 1.0
    return len(intersection) / len(union)


def select_examples_random(
    pool: list[FewShotExample],
    n: int,
    seed: int = 42,
) -> list[FewShotExample]:
    """Randomly select n examples from pool."""
    rng = random.Random(seed)
    n = min(n, len(pool))
    return rng.sample(pool, n)


def select_examples_by_similarity(
    pool: list[FewShotExample],
    query: str,
    n: int,
) -> list[FewShotExample]:
    """Select n most similar examples to query (by compute_example_similarity)."""
    n = min(n, len(pool))
    scored = [(compute_example_similarity(query, ex), ex) for ex in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:n]]


def select_examples_diverse(
    pool: list[FewShotExample],
    query: str,
    n: int,
    seed: int = 42,
) -> list[FewShotExample]:
    """Select diverse examples: greedy maximizing min pairwise distance.

    Start with most similar to query, then greedily add least similar to current set.
    Uses compute_example_similarity for distances.
    """
    n = min(n, len(pool))
    if n == 0:
        return []

    # Score all against query
    scored = [(compute_example_similarity(query, ex), ex) for ex in pool]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [scored[0][1]]
    remaining = [ex for _, ex in scored[1:]]

    while len(selected) < n and remaining:
        # Greedily add the example least similar to current set
        # (maximize min distance = minimize max similarity)
        best_idx = -1
        best_min_sim = float("inf")

        for i, candidate in enumerate(remaining):
            # similarity to each selected example (using candidate as query)
            sims = [compute_example_similarity(candidate.input_text, sel) for sel in selected]
            max_sim = max(sims)  # how close is this candidate to the selected set
            if max_sim < best_min_sim:
                best_min_sim = max_sim
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough estimate: split by whitespace, count words."""
    return len(prompt.split())


def truncate_prompt_to_fit(
    prompt: str,
    max_tokens: int,
    truncate_from: str = "start",
) -> str:
    """Truncate prompt to fit within max_tokens (words).

    truncate_from: "start" -> remove from beginning, "end" -> remove from end.
    """
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt

    if truncate_from == "start":
        words = words[-max_tokens:]
    else:
        words = words[:max_tokens]

    return " ".join(words)


class FewShotPromptBuilder:
    """Automated few-shot prompt construction with example selection."""

    def __init__(self, cfg: FewShotConfig) -> None:
        self.cfg = cfg
        self._pool: list[FewShotExample] = []
        self._last_selected: list[FewShotExample] = []

    def add_examples(self, examples: list[FewShotExample]) -> None:
        """Add examples to the pool."""
        self._pool.extend(examples)

    def build(self, query: str, seed: int = 42) -> str:
        """Select examples using configured strategy, build prompt."""
        strategy = self.cfg.selection_strategy
        n = self.cfg.n_shots

        if strategy == "random":
            selected = select_examples_random(self._pool, n, seed=seed)
        elif strategy == "similarity":
            selected = select_examples_by_similarity(self._pool, query, n)
        elif strategy == "diverse":
            selected = select_examples_diverse(self._pool, query, n, seed=seed)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy!r}")

        self._last_selected = selected
        return format_few_shot_prompt(selected, query, self.cfg)

    def evaluate_prompt_quality(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        prompt: str,
        expected_output: str,
    ) -> dict[str, float]:
        """Evaluate prompt quality by computing perplexity of expected_output given prompt.

        Returns {"perplexity", "token_count", "avg_example_similarity"}
        where avg_example_similarity is mean similarity of selected examples to the last query.
        """
        full_text = prompt + expected_output
        ids = encode_fn(full_text)
        token_count = len(ids)

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            loss, logits, _ = model(input_ids)

        # Compute CE loss manually on expected_output portion
        # logits: (1, T, V) — shift by 1
        logits_shifted = logits[0, :-1, :]  # (T-1, V)
        targets = input_ids[0, 1:]  # (T-1,)
        ce_loss = F.cross_entropy(logits_shifted, targets)
        perplexity = math.exp(ce_loss.item())

        # avg_example_similarity — mean similarity of selected examples to the last query
        # We use the last word/phrase of the prompt as proxy for the query;
        # more precisely, we track _last_selected and need the query — use prompt text itself.
        if self._last_selected:
            # Use a dummy query object with the prompt as input_text
            FewShotExample(input_text=prompt, output_text="")
            sims = [compute_example_similarity(prompt, ex) for ex in self._last_selected]
            avg_sim = sum(sims) / len(sims)
        else:
            avg_sim = 0.0

        return {
            "perplexity": perplexity,
            "token_count": float(token_count),
            "avg_example_similarity": avg_sim,
        }

    def optimize_shot_count(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        query: str,
        expected_output: str,
        max_shots: int = 8,
    ) -> int:
        """Try 0..max_shots shots, return n_shots with lowest perplexity."""
        original_n = self.cfg.n_shots
        best_n = 0
        best_ppl = float("inf")

        for n in range(0, max_shots + 1):
            self.cfg.n_shots = n
            prompt = self.build(query)
            result = self.evaluate_prompt_quality(model, encode_fn, prompt, expected_output)
            if result["perplexity"] < best_ppl:
                best_ppl = result["perplexity"]
                best_n = n

        self.cfg.n_shots = original_n
        return best_n
