"""Symbolic reasoning evaluation: syllogisms, arithmetic, boolean logic, and set problems."""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class LogicProblem:
    """A single symbolic reasoning problem."""

    problem_type: str  # "syllogism" | "arithmetic" | "boolean" | "set"
    premises: list[str]
    question: str
    answer: str
    difficulty: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_name(rng: random.Random, length: int = 3) -> str:
    """Return a random lowercase alphabetic string of given length."""
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _unique_names(rng: random.Random, n: int, length: int = 3) -> list[str]:
    """Return n unique random names."""
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        name = _rand_name(rng, length)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_syllogism(rng: random.Random = None) -> LogicProblem:
    """Generate a random syllogism problem.

    Template: "All {A} are {B}. All {B} are {C}. Are all {A} {C}? Answer: yes"
    Uses random 3-char alphabetic entity names.
    """
    if rng is None:
        rng = random.Random()  # noqa: S311

    A, B, C = _unique_names(rng, 3)

    premises = [
        f"All {A} are {B}.",
        f"All {B} are {C}.",
    ]
    question = f"Are all {A} {C}?"
    answer = "yes"

    return LogicProblem(
        problem_type="syllogism",
        premises=premises,
        question=question,
        answer=answer,
        difficulty=1,
    )


def generate_arithmetic(rng: random.Random = None, max_val: int = 20) -> LogicProblem:
    """Generate a multi-step arithmetic problem.

    Problem: "If x={a} and y={b}, what is x+y*2?"
    Answer is the integer result as a string.
    """
    if rng is None:
        rng = random.Random()  # noqa: S311

    a = rng.randint(1, max_val)
    b = rng.randint(1, max_val)
    result = a + b * 2

    premises = [f"x = {a}", f"y = {b}"]
    question = "What is x + y * 2?"
    answer = str(result)

    return LogicProblem(
        problem_type="arithmetic",
        premises=premises,
        question=question,
        answer=answer,
        difficulty=1,
    )


def generate_boolean_logic(rng: random.Random = None) -> LogicProblem:
    """Generate a boolean logic expression problem.

    Uses 3-5 boolean operations randomly (AND, OR).
    Returns answer as "True" or "False".
    """
    if rng is None:
        rng = random.Random()  # noqa: S311

    n_ops = rng.randint(3, 5)
    ops = [rng.choice(["AND", "OR"]) for _ in range(n_ops - 1)]
    values = [rng.choice([True, False]) for _ in range(n_ops)]

    value_strs = ["True" if v else "False" for v in values]

    parts = [value_strs[0]]
    for i, op in enumerate(ops):
        parts.append(op)
        parts.append(value_strs[i + 1])
    expression = " ".join(parts)

    # Evaluate left-to-right
    result = values[0]
    for i, op in enumerate(ops):
        if op == "AND":
            result = result and values[i + 1]
        else:
            result = result or values[i + 1]

    answer = "True" if result else "False"

    return LogicProblem(
        problem_type="boolean",
        premises=[f"Evaluate: {expression}"],
        question=f"What is {expression}?",
        answer=answer,
        difficulty=1,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def evaluate_answer(predicted: str, gold: str, problem_type: str) -> float:
    """Score a predicted answer against the gold answer.

    - Exact string match after strip/lower -> 1.0
    - For arithmetic: int comparison -> 1.0 if equal, 0.0 if not
    - For boolean: "yes"/"true" and "no"/"false" treated as equivalent
    - Otherwise: 0.0
    """
    pred_clean = predicted.strip().lower()
    gold_clean = gold.strip().lower()

    # Exact match
    if pred_clean == gold_clean:
        return 1.0

    # Arithmetic: try integer comparison
    if problem_type == "arithmetic":
        try:
            return 1.0 if int(pred_clean) == int(gold_clean) else 0.0
        except (ValueError, TypeError):
            return 0.0

    # Boolean: treat yes/true and no/false as equivalent
    if problem_type in ("boolean", "syllogism"):
        truthy = {"yes", "true"}
        falsy = {"no", "false"}
        if pred_clean in truthy and gold_clean in truthy:
            return 1.0
        if pred_clean in falsy and gold_clean in falsy:
            return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Greedy decode helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def _greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy autoregressive decode; returns generated token ids (excluding prompt)."""
    generated = []
    past_key_values = None

    _, logits, past_key_values = model(input_ids, past_key_values=past_key_values)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        _, logits, past_key_values = model(next_token, past_key_values=past_key_values)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=1)


# ---------------------------------------------------------------------------
# SymbolicReasoningBenchmark
# ---------------------------------------------------------------------------


class SymbolicReasoningBenchmark:
    """Benchmark for symbolic reasoning tasks."""

    def __init__(self, n_problems: int = 50, seed: int = 42) -> None:
        self.n_problems = n_problems
        self.seed = seed
        self._rng = random.Random(seed)  # noqa: S311
        self.problems: list[LogicProblem] = self._generate_problems()

    def _generate_problems(self) -> list[LogicProblem]:
        """Generate n_problems evenly distributed across problem types."""
        generators = [
            generate_syllogism,
            generate_arithmetic,
            generate_boolean_logic,
        ]
        problems: list[LogicProblem] = []
        for i in range(self.n_problems):
            gen = generators[i % len(generators)]
            problems.append(gen(rng=self._rng))
        return problems

    @staticmethod
    def generate_prompt(problem: LogicProblem) -> str:
        """Format a problem into a prompt string.

        Format: "{premises joined by newlines}\\nQuestion: {question}\\nAnswer:"
        """
        premises_str = "\n".join(problem.premises)
        return f"{premises_str}\nQuestion: {problem.question}\nAnswer:"

    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        max_new_tokens: int = 16,
    ) -> dict:
        """Evaluate model on all problems.

        Returns:
            {
                "accuracy": float,
                "by_type": dict[str, float],
                "n_problems": int,
            }
        """
        model.eval()
        device = next(model.parameters()).device

        scores: list[float] = []
        by_type: dict[str, list[float]] = {}

        for problem in self.problems:
            prompt = self.generate_prompt(problem)
            input_ids = torch.tensor([tokenizer_encode(prompt)], dtype=torch.long, device=device)
            gen_ids = _greedy_decode(model, input_ids, max_new_tokens)
            generated_text = tokenizer_decode(gen_ids[0].tolist())

            score = evaluate_answer(generated_text, problem.answer, problem.problem_type)
            scores.append(score)
            by_type.setdefault(problem.problem_type, []).append(score)

        accuracy = sum(scores) / len(scores) if scores else 0.0
        type_acc = {t: sum(v) / len(v) for t, v in by_type.items()}

        return {
            "accuracy": accuracy,
            "by_type": type_acc,
            "n_problems": len(self.problems),
        }
