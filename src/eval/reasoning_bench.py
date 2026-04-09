"""Synthetic reasoning benchmarks: arithmetic word problems, logical deduction, and pattern completion."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the reasoning benchmark suite."""

    n_problems: int = 100
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    problem_types: list[str] = field(
        default_factory=lambda: ["arithmetic", "logic", "analogy", "sequence"]
    )
    seed: int = 42
    max_answer_len: int = 32


@dataclass
class Problem:
    """A single benchmark problem."""

    type: str
    prompt: str
    answer: str
    difficulty: str
    metadata: dict


# ---------------------------------------------------------------------------
# Arithmetic problem generators
# ---------------------------------------------------------------------------

def generate_arithmetic_problem(difficulty: str, rng: random.Random) -> Problem:
    """Generate an arithmetic word problem.

    Easy: single-digit addition/subtraction.
    Medium: two-step problem with multiplication (A * B + C).
    Hard: three-step problem with fractions (A/B * C - D).
    """
    if difficulty == "easy":
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        result = a + b if op == "+" else a - b
        prompt = f"What is {a} {op} {b}? Answer with the number only."
        answer = str(result)
        metadata = {"a": a, "b": b, "op": op}

    elif difficulty == "medium":
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        c = rng.randint(1, 20)
        result = a * b + c
        prompt = (
            f"If you have {a} groups of {b} items and then add {c} more, "
            f"how many items do you have in total? Answer with the number only."
        )
        answer = str(result)
        metadata = {"a": a, "b": b, "c": c, "op": "mul_add"}

    else:  # hard
        b = rng.randint(2, 6)
        a = rng.randint(1, b - 1)  # proper fraction a/b
        # Make c a multiple of b so the result is an integer
        multiplier = rng.randint(1, 5)
        c = b * multiplier
        d = rng.randint(1, 10)
        frac_result = Fraction(a, b) * c - d
        result = int(frac_result) if frac_result.denominator == 1 else float(frac_result)
        prompt = (
            f"Calculate: ({a}/{b}) * {c} - {d}. Answer with the number only."
        )
        answer = str(result)
        metadata = {"a": a, "b": b, "c": c, "d": d, "op": "frac_mul_sub"}

    return Problem(
        type="arithmetic",
        prompt=prompt,
        answer=answer,
        difficulty=difficulty,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Logic problem generators
# ---------------------------------------------------------------------------

_SYLLOGISM_TEMPLATES = [
    ("mammals", "dogs", "warm-blooded"),
    ("birds", "eagles", "feathered"),
    ("plants", "roses", "living organisms"),
    ("metals", "gold", "conductive"),
    ("primates", "chimpanzees", "intelligent"),
]

_DEDUCTION_CHAINS = [
    {
        "premises": ["All A are B.", "All B are C.", "All C are D."],
        "question": "Is A D?",
        "answer": "Yes",
    },
    {
        "premises": ["All X are Y.", "All Y are Z.", "No Z are W."],
        "question": "Can X be W?",
        "answer": "No",
    },
    {
        "premises": ["If P then Q.", "If Q then R.", "P is true."],
        "question": "Is R true?",
        "answer": "Yes",
    },
]


def generate_logic_problem(difficulty: str, rng: random.Random) -> Problem:
    """Generate a logical deduction problem.

    Easy: Boolean AND/OR evaluation.
    Medium: Syllogism (All X are Y. Z is X. Is Z Y?).
    Hard: Multi-step deduction chain (3+ steps).
    """
    if difficulty == "easy":
        a_val = rng.choice([True, False])
        b_val = rng.choice([True, False])
        op = rng.choice(["AND", "OR"])
        result = (a_val and b_val) if op == "AND" else (a_val or b_val)
        a_str = "true" if a_val else "false"
        b_str = "true" if b_val else "false"
        prompt = (
            f"If A is {a_str} and B is {b_str}, is A {op} B true or false? "
            f"Answer True or False."
        )
        answer = "True" if result else "False"
        metadata = {"a": a_val, "b": b_val, "op": op}

    elif difficulty == "medium":
        tmpl = rng.choice(_SYLLOGISM_TEMPLATES)
        category, member, prop = tmpl
        prompt = (
            f"All {category} are {prop}. {member.capitalize()} are {category}. "
            f"Are {member} {prop}? Answer Yes or No."
        )
        answer = "Yes"
        metadata = {"category": category, "member": member, "property": prop}

    else:  # hard
        chain = rng.choice(_DEDUCTION_CHAINS)
        premises_str = " ".join(chain["premises"])
        prompt = f"{premises_str} {chain['question']} Answer Yes or No."
        answer = chain["answer"]
        metadata = {"premises": chain["premises"], "question": chain["question"]}

    return Problem(
        type="logic",
        prompt=prompt,
        answer=answer,
        difficulty=difficulty,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Analogy problem generators
# ---------------------------------------------------------------------------

_ANALOGY_CATEGORIES: dict[str, list[tuple]] = {
    "color_shade": [
        ("red", "crimson", "blue", "navy", ["navy", "azure", "sapphire", "cobalt"]),
        ("green", "emerald", "yellow", "gold", ["gold", "amber", "lemon", "citrine"]),
        ("white", "ivory", "black", "ebony", ["ebony", "onyx", "jet", "coal"]),
    ],
    "animal_sound": [
        ("dog", "bark", "cat", "meow", ["meow", "purr", "hiss", "chirp"]),
        ("cow", "moo", "duck", "quack", ["quack", "honk", "cluck", "chirp"]),
        ("lion", "roar", "snake", "hiss", ["hiss", "rattle", "slither", "whistle"]),
    ],
    "country_capital": [
        ("France", "Paris", "Germany", "Berlin", ["Berlin", "Munich", "Hamburg", "Frankfurt"]),
        ("Japan", "Tokyo", "China", "Beijing", ["Beijing", "Shanghai", "Guangzhou", "Nanjing"]),
        ("USA", "Washington", "UK", "London", ["London", "Manchester", "Edinburgh", "Liverpool"]),
    ],
}


def generate_analogy_problem(difficulty: str, rng: random.Random) -> Problem:
    """Generate an analogy problem: A is to B as C is to ?

    Uses category relationships: color->shade, animal->sound, country->capital.
    Multiple choices are stored in metadata.
    """
    category = rng.choice(list(_ANALOGY_CATEGORIES.keys()))
    items = _ANALOGY_CATEGORIES[category]
    a, b, c, correct, choices = rng.choice(items)

    shuffled = choices[:]
    rng.shuffle(shuffled)

    choices_str = ", ".join(shuffled)
    prompt = (
        f"{a} is to {b} as {c} is to ? "
        f"Choose one: {choices_str}."
    )

    return Problem(
        type="analogy",
        prompt=prompt,
        answer=correct,
        difficulty=difficulty,
        metadata={
            "a": a,
            "b": b,
            "c": c,
            "choices": shuffled,
            "category": category,
        },
    )


# ---------------------------------------------------------------------------
# Sequence problem generators
# ---------------------------------------------------------------------------

def generate_sequence_problem(difficulty: str, rng: random.Random) -> Problem:
    """Generate a number sequence completion problem.

    Easy: arithmetic sequence (next value).
    Medium: geometric sequence (next value).
    Hard: Fibonacci-style sequence (next value).
    """
    if difficulty == "easy":
        start = rng.randint(0, 10)
        step = rng.randint(1, 5)
        length = 4
        seq = [start + i * step for i in range(length)]
        next_val = start + length * step
        seq_str = ", ".join(str(x) for x in seq)
        prompt = f"What comes next in the sequence: {seq_str}, ? Answer with the number only."
        answer = str(next_val)
        metadata = {"start": start, "step": step, "type": "arithmetic", "sequence": seq}

    elif difficulty == "medium":
        start = rng.randint(1, 4)
        ratio = rng.choice([2, 3])
        length = 4
        seq = [start * (ratio ** i) for i in range(length)]
        next_val = start * (ratio ** length)
        seq_str = ", ".join(str(x) for x in seq)
        prompt = f"What comes next in the sequence: {seq_str}, ? Answer with the number only."
        answer = str(next_val)
        metadata = {"start": start, "ratio": ratio, "type": "geometric", "sequence": seq}

    else:  # hard - Fibonacci-style
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        shown = seq[:6]
        next_val = seq[6]
        seq_str = ", ".join(str(x) for x in shown)
        prompt = (
            f"What comes next in the Fibonacci-like sequence: {seq_str}, ? "
            f"Answer with the number only."
        )
        answer = str(next_val)
        metadata = {"a": a, "b": b, "type": "fibonacci", "sequence": shown}

    return Problem(
        type="sequence",
        prompt=prompt,
        answer=answer,
        difficulty=difficulty,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_final_answer(text: str) -> str:
    """Extract the last numeric or Yes/No/True/False token from generated text."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    yn = re.findall(r"\b(yes|no|true|false)\b", text.lower())
    if numbers:
        return numbers[-1]
    if yn:
        return yn[-1].capitalize()
    words = text.strip().split()
    return words[-1] if words else ""


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
# ReasoningBenchmark
# ---------------------------------------------------------------------------

_GENERATOR_MAP: dict[str, Callable] = {
    "arithmetic": generate_arithmetic_problem,
    "logic": generate_logic_problem,
    "analogy": generate_analogy_problem,
    "sequence": generate_sequence_problem,
}

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class ReasoningBenchmark:
    """Full reasoning benchmark suite."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self.problems: list[Problem] = self.generate_problems()

    def generate_problems(self) -> list[Problem]:
        """Generate n_problems evenly distributed across problem_types."""
        types = self.config.problem_types
        n = self.config.n_problems
        difficulty = self.config.difficulty

        problems: list[Problem] = []
        per_type = n // len(types)
        remainder = n % len(types)

        for i, ptype in enumerate(types):
            count = per_type + (1 if i < remainder else 0)
            generator = _GENERATOR_MAP.get(ptype)
            if generator is None:
                raise ValueError(f"Unknown problem type: {ptype!r}")
            for _ in range(count):
                problems.append(generator(difficulty, self._rng))

        problems.sort(key=lambda p: _DIFFICULTY_ORDER.get(p.difficulty, 1))
        return problems

    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        max_new_tokens: int = 32,
    ) -> dict[str, Any]:
        """Evaluate model on all problems via greedy decoding + exact match.

        Returns:
            {
                "overall_accuracy": float,
                "by_type": dict[str, float],
                "by_difficulty": dict[str, float],
            }
        """
        model.eval()
        device = next(model.parameters()).device

        correct_total = 0
        by_type: dict[str, list[bool]] = {}
        by_difficulty: dict[str, list[bool]] = {}

        for problem in self.problems:
            input_ids = torch.tensor(
                [tokenizer_encode(problem.prompt)], dtype=torch.long, device=device
            )
            gen_ids = _greedy_decode(model, input_ids, max_new_tokens)
            generated_text = tokenizer_decode(gen_ids[0].tolist())

            pred = _normalize(_extract_final_answer(generated_text))
            gold = _normalize(problem.answer)
            is_correct = pred == gold

            correct_total += int(is_correct)
            by_type.setdefault(problem.type, []).append(is_correct)
            by_difficulty.setdefault(problem.difficulty, []).append(is_correct)

        n = len(self.problems)
        overall = correct_total / n if n > 0 else 0.0
        type_acc = {t: sum(v) / len(v) for t, v in by_type.items()}
        diff_acc = {d: sum(v) / len(v) for d, v in by_difficulty.items()}

        return {
            "overall_accuracy": overall,
            "by_type": type_acc,
            "by_difficulty": diff_acc,
        }

    def format_few_shot(self, problem: Problem, n_shots: int = 3) -> str:
        """Format a problem with n_shots example problems prepended."""
        candidates = [
            p for p in self.problems
            if p is not problem and p.type == problem.type
        ]
        if len(candidates) < n_shots:
            candidates = [p for p in self.problems if p is not problem]

        shots = candidates[:n_shots]
        lines: list[str] = []
        for ex in shots:
            lines.append(f"Q: {ex.prompt}")
            lines.append(f"A: {ex.answer}")
            lines.append("")

        lines.append(f"Q: {problem.prompt}")
        lines.append("A:")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chain-of-thought evaluation
# ---------------------------------------------------------------------------

def evaluate_chain_of_thought(
    model: nn.Module,
    problems: list[Problem],
    tokenizer_encode: Callable,
    tokenizer_decode: Callable,
    max_new_tokens: int = 64,
) -> dict[str, float]:
    """Evaluate with and without CoT prompting.

    CoT prompt: "Think step by step. " prepended to each problem prompt.

    Returns:
        {
            "cot_accuracy": float,
            "direct_accuracy": float,
            "cot_improvement": float,
        }
    """
    model.eval()
    device = next(model.parameters()).device

    cot_correct = 0
    direct_correct = 0
    n = len(problems)

    for problem in problems:
        gold = _normalize(problem.answer)

        # Direct
        direct_ids = torch.tensor(
            [tokenizer_encode(problem.prompt)], dtype=torch.long, device=device
        )
        direct_gen = _greedy_decode(model, direct_ids, min(max_new_tokens, 32))
        direct_text = tokenizer_decode(direct_gen[0].tolist())
        direct_pred = _normalize(_extract_final_answer(direct_text))
        direct_correct += int(direct_pred == gold)

        # CoT
        cot_prompt = "Think step by step. " + problem.prompt
        cot_ids = torch.tensor(
            [tokenizer_encode(cot_prompt)], dtype=torch.long, device=device
        )
        cot_gen = _greedy_decode(model, cot_ids, max_new_tokens)
        cot_text = tokenizer_decode(cot_gen[0].tolist())
        cot_pred = _normalize(_extract_final_answer(cot_text))
        cot_correct += int(cot_pred == gold)

    cot_acc = cot_correct / n if n > 0 else 0.0
    direct_acc = direct_correct / n if n > 0 else 0.0

    return {
        "cot_accuracy": cot_acc,
        "direct_accuracy": direct_acc,
        "cot_improvement": cot_acc - direct_acc,
    }
