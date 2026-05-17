"""AMC-specific memory benchmark scaffold.

This benchmark targets the Aurelian Memory Core (AMC) rather than general
language ability.  It is intentionally deterministic and dependency-free so it
can run in unit tests, CI, and early architecture experiments before a full
model checkpoint exists.

The benchmark evaluates four memory behaviors:

* cross_session_recall: retrieve a value from a prior session transcript.
* surprise_gate_selectivity: choose the observation that should be stored.
* consolidation_preference: prefer repeated, high-confidence memories.
* contradiction_quarantine: detect conflicting memories that need verification.

A model runner only needs to supply ``generate_fn(prompt) -> str``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class AMCMemoryExample:
    """Single AMC benchmark example."""

    task: str
    prompt: str
    expected: str
    seed: int
    context_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AMCMemoryBenchmark:
    """Deterministic benchmark for memory-specific AMC behavior."""

    TASKS: tuple[str, ...] = (
        "cross_session_recall",
        "surprise_gate_selectivity",
        "consolidation_preference",
        "contradiction_quarantine",
    )

    _FACTS: tuple[tuple[str, str], ...] = (
        ("deployment_default_host", "127.0.0.1"),
        ("canonical_architecture", "AMC-first focused build"),
        ("alignment_scope", "SFT DPO GRPO constitutional-memory quarantine"),
        ("benchmark_gate", "RULER plus AMC-Memory"),
        ("memory_tier_three", "long-term consolidated memory"),
    )

    _OBSERVATIONS: tuple[tuple[str, str, int], ...] = (
        ("obs_a", "The user says hello and asks a transient greeting question.", 1),
        (
            "obs_b",
            "The user corrects the architecture: Aurelius must stay AMC-first.",
            9,
        ),
        ("obs_c", "A tool returns a temporary progress percentage of 41%.", 2),
        ("obs_d", "A generated joke receives no follow-up or correction.", 1),
    )

    def __init__(self, approx_chars_per_token: int = APPROX_CHARS_PER_TOKEN) -> None:
        if not isinstance(approx_chars_per_token, int) or approx_chars_per_token <= 0:
            raise ValueError(
                f"approx_chars_per_token must be a positive int, got {approx_chars_per_token!r}"
            )
        self.approx_chars_per_token = approx_chars_per_token

    def _char_budget(self, context_tokens: int) -> int:
        if not isinstance(context_tokens, int) or context_tokens <= 0:
            raise ValueError(f"context_tokens must be a positive int, got {context_tokens!r}")
        return context_tokens * self.approx_chars_per_token

    @staticmethod
    def _filler(char_budget: int, rng: random.Random) -> str:
        sentences = [
            "Routine telemetry was recorded for the prior request.",
            "A short note described unrelated UI rendering behavior.",
            "The assistant summarized a build log with no durable preference.",
            "A temporary cache key was mentioned and then expired.",
            "The session contained ordinary conversational filler.",
            "No policy change was approved in this paragraph.",
        ]
        rng.shuffle(sentences)
        out: list[str] = []
        total = 0
        i = 0
        while total < char_budget:
            sentence = sentences[i % len(sentences)]
            out.append(sentence)
            total += len(sentence) + 1
            i += 1
        return " ".join(out)[:char_budget]

    def build_cross_session_recall(
        self, context_tokens: int = 512, seed: int = 0
    ) -> AMCMemoryExample:
        """Build a task that requires recalling one fact from prior session text."""
        rng = random.Random(seed)  # noqa: S311 - deterministic benchmark generation
        facts = list(self._FACTS)
        rng.shuffle(facts)
        target_key, target_value = facts[0]
        memory_lines = [f"MEMORY[{key}] = {value}." for key, value in facts]
        filler = self._filler(self._char_budget(context_tokens), rng)
        prompt = (
            "You are evaluating AMC cross-session recall.\n"
            "Prior session transcript:\n"
            f"{filler}\n"
            + "\n".join(memory_lines)
            + "\n\nQuestion: What is the exact value stored for "
            f"{target_key}? Return only the value."
        )
        return AMCMemoryExample(
            task="cross_session_recall",
            prompt=prompt,
            expected=target_value,
            seed=seed,
            context_tokens=context_tokens,
            metadata={"target_key": target_key},
        )

    def build_surprise_gate_selectivity(
        self, context_tokens: int = 512, seed: int = 0
    ) -> AMCMemoryExample:
        """Build a task that selects the most memory-worthy observation."""
        rng = random.Random(seed)  # noqa: S311 - deterministic benchmark generation
        observations = list(self._OBSERVATIONS)
        rng.shuffle(observations)
        best = max(observations, key=lambda item: item[2])
        filler = self._filler(max(64, self._char_budget(context_tokens) // 3), rng)
        rows = [f"{obs_id}: importance={score}; {text}" for obs_id, text, score in observations]
        prompt = (
            "You are evaluating AMC surprise-gate selectivity.\n"
            "Choose the single observation id that should be written to "
            "episodic memory. Prefer durable corrections, architecture "
            "decisions, and high-surprise information.\n"
            f"Context filler: {filler}\n"
            + "\n".join(rows)
            + "\nAnswer with only the observation id."
        )
        return AMCMemoryExample(
            task="surprise_gate_selectivity",
            prompt=prompt,
            expected=best[0],
            seed=seed,
            context_tokens=context_tokens,
            metadata={"best_score": best[2]},
        )

    def build_consolidation_preference(
        self, context_tokens: int = 512, seed: int = 0
    ) -> AMCMemoryExample:
        """Build a task that prefers repeated/high-confidence memory."""
        rng = random.Random(seed)  # noqa: S311 - deterministic benchmark generation
        durable = "Aurelius should prioritize AMC ablations before Atlas-scale expansion"
        distractors = [
            "A temporary port number was 49152 during one local run",
            "A draft UI color was violet in an exploratory screenshot",
            "A single failed command produced a transient timeout",
        ]
        evidence = [
            f"session_1 confidence=0.82: {durable}.",
            f"session_2 confidence=0.88: {durable}.",
            f"session_3 confidence=0.91: {durable}.",
        ] + [f"session_x confidence=0.40: {d}." for d in distractors]
        rng.shuffle(evidence)
        filler = self._filler(self._char_budget(context_tokens), rng)
        prompt = (
            "You are evaluating AMC consolidation preference.\n"
            "Pick the memory that should be promoted to long-term storage. "
            "Prefer repeated, high-confidence, architecture-level facts over "
            "one-off transient details.\n"
            f"Background: {filler}\n"
            + "\n".join(evidence)
            + "\nReturn the durable memory text only."
        )
        return AMCMemoryExample(
            task="consolidation_preference",
            prompt=prompt,
            expected=durable,
            seed=seed,
            context_tokens=context_tokens,
            metadata={"repetitions": 3},
        )

    def build_contradiction_quarantine(
        self, context_tokens: int = 512, seed: int = 0
    ) -> AMCMemoryExample:
        """Build a task that requires quarantining contradictory memory."""
        rng = random.Random(seed)  # noqa: S311 - deterministic benchmark generation
        filler = self._filler(self._char_budget(context_tokens), rng)
        prompt = (
            "You are evaluating AMC memory safety.\n"
            "Two candidate long-term memories conflict:\n"
            "memory_a confidence=0.64 source=tool: The production server "
            "default host is 0.0.0.0.\n"
            "memory_b confidence=0.93 source=security-review: The production "
            "server default host must be 127.0.0.1 unless explicitly exposed.\n"
            f"Additional unrelated context: {filler}\n"
            "Question: What should AMC do before using these conflicting "
            "memories? Answer with one word: use, ignore, or quarantine."
        )
        return AMCMemoryExample(
            task="contradiction_quarantine",
            prompt=prompt,
            expected="quarantine",
            seed=seed,
            context_tokens=context_tokens,
            metadata={"conflict": "host_default"},
        )

    def _build_task(self, task: str, context_tokens: int, seed: int) -> AMCMemoryExample:
        if task == "cross_session_recall":
            return self.build_cross_session_recall(context_tokens, seed)
        if task == "surprise_gate_selectivity":
            return self.build_surprise_gate_selectivity(context_tokens, seed)
        if task == "consolidation_preference":
            return self.build_consolidation_preference(context_tokens, seed)
        if task == "contradiction_quarantine":
            return self.build_contradiction_quarantine(context_tokens, seed)
        raise ValueError(f"unknown task {task!r}; known tasks: {self.TASKS}")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _check(cls, expected: str, response: str) -> bool:
        expected_norm = cls._normalize(expected)
        response_norm = cls._normalize(response)
        return expected_norm == response_norm

    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        tasks: Sequence[str] | None = None,
        context_tokens: int = 512,
        samples_per: int = 3,
    ) -> dict[str, dict[str, object]]:
        """Run benchmark tasks and return per-task pass rates and cells."""
        if not callable(generate_fn):
            raise ValueError("generate_fn must be callable")
        selected_tasks = list(self.TASKS if tasks is None else tasks)
        if not selected_tasks:
            raise ValueError("tasks must be non-empty")
        if not isinstance(samples_per, int) or samples_per <= 0:
            raise ValueError(f"samples_per must be a positive int, got {samples_per!r}")
        for task in selected_tasks:
            if task not in self.TASKS:
                raise ValueError(f"unknown task {task!r}; known tasks: {self.TASKS}")
        self._char_budget(context_tokens)

        results: dict[str, dict[str, object]] = {}
        for task in selected_tasks:
            cells: list[dict[str, object]] = []
            for seed in range(samples_per):
                example = self._build_task(task, context_tokens, seed)
                response = generate_fn(example.prompt)
                passed = self._check(example.expected, response)
                cells.append(
                    {
                        "task": task,
                        "context_tokens": context_tokens,
                        "seed": seed,
                        "expected": example.expected,
                        "response": response,
                        "pass": passed,
                        "metadata": example.metadata,
                    }
                )
            n_cells = len(cells)
            passes = sum(1 for cell in cells if cell["pass"])
            results[task] = {
                "pass_rate": passes / n_cells if n_cells else 0.0,
                "n": n_cells,
                "cells": cells,
            }
        return results

    @staticmethod
    def score_per_task(results: dict[str, dict[str, object]]) -> dict[str, float]:
        """Extract validated pass rates from an evaluation result."""
        scores: dict[str, float] = {}
        for task, info in results.items():
            rate = info.get("pass_rate", 0.0)
            if not isinstance(rate, int | float):
                raise TypeError(
                    f"pass_rate for {task!r} must be numeric, got {type(rate).__name__}"
                )
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"pass_rate for {task!r} out of range: {rate!r}")
            scores[task] = float(rate)
        return scores

    @classmethod
    def overall_score(cls, results: dict[str, dict[str, object]]) -> float:
        """Mean per-task pass rate."""
        scores = cls.score_per_task(results)
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)
