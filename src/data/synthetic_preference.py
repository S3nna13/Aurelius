"""Synthetic preference data generation: instruction-following pairs, critique augmentation, and DPO formatting."""  # noqa: E501

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreferenceDataConfig:
    """Configuration for synthetic preference data generation."""

    n_pairs: int = 100
    seed: int = 42
    augment_with_critique: bool = True
    quality_threshold: float = 0.5
    domains: list[str] = field(default_factory=lambda: ["math", "coding", "reasoning"])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A single chosen/rejected preference pair for RLHF/DPO training."""

    prompt: str
    chosen: str
    rejected: str
    domain: str
    score_chosen: float
    score_rejected: float
    critique: str | None = None


# ---------------------------------------------------------------------------
# Domain pair generators
# ---------------------------------------------------------------------------


def generate_math_pair(rng: random.Random) -> PreferencePair:
    """Generate a math question with a correct and an incorrect answer.

    Correct answer is the actual computation result; incorrect answer
    is deliberately wrong (off by a small constant).
    """
    _OPERATIONS = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    a = rng.randint(1, 50)
    b = rng.randint(1, 50)
    op_symbol, op_fn = rng.choice(_OPERATIONS)
    correct_result = op_fn(a, b)
    wrong_result = correct_result + rng.choice([-2, -1, 1, 2, 3])

    prompt = f"What is {a} {op_symbol} {b}?"
    chosen = f"{a} {op_symbol} {b} = {correct_result}"
    rejected = f"{a} {op_symbol} {b} = {wrong_result}"

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        domain="math",
        score_chosen=1.0,
        score_rejected=0.0,
    )


def generate_coding_pair(rng: random.Random) -> PreferencePair:
    """Generate a coding question with a correct Python snippet and a buggy variant.

    The buggy variant has either a wrong variable name or an off-by-one error.
    """
    _TASKS = [
        (
            "Write a Python function that returns the sum of a list.",
            "def sum_list(nums):\n    return sum(nums)",
            "def sum_list(nums):\n    return sum(num)",  # wrong variable name
        ),
        (
            "Write a Python function that returns the length of a string.",
            "def string_length(s):\n    return len(s)",
            "def string_length(s):\n    return len(s) - 1",  # off-by-one
        ),
        (
            "Write a Python function that reverses a list.",
            "def reverse_list(lst):\n    return lst[::-1]",
            "def reverse_list(lst):\n    return lst[::1]",  # wrong step
        ),
        (
            "Write a Python function that returns the maximum value in a list.",
            "def max_value(nums):\n    return max(nums)",
            "def max_value(nums):\n    return min(nums)",  # wrong function
        ),
        (
            "Write a Python function that checks if a number is even.",
            "def is_even(n):\n    return n % 2 == 0",
            "def is_even(n):\n    return n % 2 == 1",  # off-by-one in modulo check
        ),
    ]

    prompt, good_code, bad_code = rng.choice(_TASKS)

    return PreferencePair(
        prompt=prompt,
        chosen=good_code,
        rejected=bad_code,
        domain="coding",
        score_chosen=1.0,
        score_rejected=0.0,
    )


def generate_reasoning_pair(rng: random.Random) -> PreferencePair:
    """Generate a logical reasoning question with correct and incorrect reasoning."""
    _TEMPLATES = [
        # (prompt_template, correct_reasoning_fn, incorrect_reasoning_fn)
        "age_comparison",
        "sequence_next",
        "simple_arithmetic_word",
    ]

    template = rng.choice(_TEMPLATES)

    if template == "age_comparison":
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        n1, n2 = rng.sample(names, 2)
        age1 = rng.randint(20, 50)
        diff = rng.randint(1, 15)
        prompt = f"{n1} is {age1} years old. {n2} is {diff} years older than {n1}. How old is {n2}?"
        correct_age = age1 + diff
        wrong_age = age1 - diff  # subtracted instead of added
        chosen = (
            f"{n2}'s age = {n1}'s age + {diff} = {age1} + {diff} = {correct_age}. "
            f"The answer is {correct_age}."
        )
        rejected = (
            f"{n2}'s age = {n1}'s age - {diff} = {age1} - {diff} = {wrong_age}. "
            f"The answer is {wrong_age}."
        )

    elif template == "sequence_next":
        start = rng.randint(1, 10)
        step = rng.randint(2, 6)
        seq = [start + step * i for i in range(4)]
        correct_next = start + step * 4
        wrong_next = correct_next + step  # skipped one term
        seq_str = ", ".join(map(str, seq))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        chosen = (
            f"Each number increases by {step}. "
            f"{seq[-1]} + {step} = {correct_next}. The answer is {correct_next}."
        )
        rejected = (
            f"Each number increases by {step * 2}. "
            f"{seq[-1]} + {step * 2} = {wrong_next}. The answer is {wrong_next}."
        )

    else:  # simple_arithmetic_word
        total = rng.randint(10, 100)
        used = rng.randint(1, total - 1)
        remaining = total - used
        prompt = f"There are {total} apples. {used} are eaten. How many remain?"
        chosen = f"{total} - {used} = {remaining}. The answer is {remaining}."
        rejected = f"{total} + {used} = {total + used}. The answer is {total + used}."

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        domain="reasoning",
        score_chosen=1.0,
        score_rejected=0.0,
    )


# ---------------------------------------------------------------------------
# Critique augmentation
# ---------------------------------------------------------------------------

_CHOSEN_REASONS = [
    "provides the correct answer with clear steps",
    "uses the right formula and arrives at the correct result",
    "demonstrates accurate logical reasoning",
    "applies the correct operation and computes the result accurately",
    "shows valid, working code that solves the problem correctly",
]

_REJECTED_PROBLEMS = [
    "contains a computational error leading to a wrong answer",
    "applies the wrong operation and reaches an incorrect conclusion",
    "makes a logical mistake that invalidates the final answer",
    "introduces a bug that causes incorrect behavior",
    "uses an incorrect formula, yielding a wrong result",
]


def augment_with_critique(pair: PreferencePair, rng: random.Random) -> PreferencePair:
    """Add a critique field explaining why chosen is better than rejected.

    Returns a new PreferencePair with the critique field filled in.
    """
    reason = rng.choice(_CHOSEN_REASONS)
    problem = rng.choice(_REJECTED_PROBLEMS)
    critique = (
        f"The chosen response is better because it {reason}. The rejected response {problem}."
    )
    return PreferencePair(
        prompt=pair.prompt,
        chosen=pair.chosen,
        rejected=pair.rejected,
        domain=pair.domain,
        score_chosen=pair.score_chosen,
        score_rejected=pair.score_rejected,
        critique=critique,
    )


# ---------------------------------------------------------------------------
# Filtering and formatting
# ---------------------------------------------------------------------------


def filter_by_quality(pairs: list[PreferencePair], threshold: float) -> list[PreferencePair]:
    """Keep only pairs where score_chosen - score_rejected >= threshold."""
    return [p for p in pairs if p.score_chosen - p.score_rejected >= threshold]


def format_for_dpo(pair: PreferencePair) -> dict:
    """Format a PreferencePair for DPO training.

    Returns a dict with keys: prompt, chosen, rejected.
    """
    return {
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
    }


def format_for_rlhf(pair: PreferencePair) -> dict:
    """Format a PreferencePair as a prompt + completion + reward record.

    Uses chosen as the completion and score_chosen as the reward.
    Returns a dict with keys: prompt, completion, reward.
    """
    return {
        "prompt": pair.prompt,
        "completion": pair.chosen,
        "reward": pair.score_chosen,
    }


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

_DOMAIN_GENERATORS = {
    "math": generate_math_pair,
    "coding": generate_coding_pair,
    "reasoning": generate_reasoning_pair,
}


class SyntheticPreferenceGenerator:
    """High-level interface for generating synthetic preference pairs."""

    def __init__(self, config: PreferenceDataConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def generate(self, n: int | None = None) -> list[PreferencePair]:
        """Generate n (or config.n_pairs) preference pairs.

        Pairs are distributed evenly across configured domains.
        Critiques are added if config.augment_with_critique is True.
        Pairs are filtered by config.quality_threshold.
        """
        cfg = self.config
        total = n if n is not None else cfg.n_pairs

        # Only generate for domains we have generators for
        domains = [d for d in cfg.domains if d in _DOMAIN_GENERATORS]
        if not domains:
            return []

        pairs: list[PreferencePair] = []
        for i in range(total):
            domain = domains[i % len(domains)]
            gen_fn = _DOMAIN_GENERATORS[domain]
            pair = gen_fn(self._rng)
            pairs.append(pair)

        if cfg.augment_with_critique:
            pairs = [augment_with_critique(p, self._rng) for p in pairs]

        pairs = filter_by_quality(pairs, cfg.quality_threshold)

        return pairs
