"""Synthetic math problem generator for RL training.

Generates arithmetic, algebra, and combinatorics problems with exact, verifiable
answers. All answers are derivable from generation parameters — no external solver
needed. Designed to supply large quantities of math problems with ground-truth
answers for reinforcement-learning-based math reasoning training.

Pure Python — no third-party dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MathProblem:
    """One synthetic math problem with a verifiable exact answer."""

    problem_id: str
    problem_text: str            # natural-language problem statement
    answer: str                  # exact answer as string (e.g. "42", "3/7", "120")
    answer_numeric: Optional[float]  # float version if applicable, else None
    difficulty: str              # "easy" | "medium" | "hard"
    problem_type: str            # "arithmetic" | "algebra" | "combinatorics"
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticMathConfig:
    """Configuration for :class:`SyntheticMathGenerator`."""

    seed: int = 42
    difficulty_distribution: dict = field(
        default_factory=lambda: {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
    problem_types: list[str] = field(
        default_factory=lambda: ["arithmetic", "algebra", "combinatorics"]
    )
    max_number: int = 1000  # upper bound for generated integer values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of |a| and |b|."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a or 1


def _reduce_fraction(num: int, den: int) -> tuple[int, int]:
    """Return (numerator, denominator) in lowest terms with positive denominator."""
    if den < 0:
        num, den = -num, -den
    g = _gcd(abs(num), abs(den))
    return num // g, den // g


def _fraction_str(num: int, den: int) -> str:
    num, den = _reduce_fraction(num, den)
    if den == 1:
        return str(num)
    return f"{num}/{den}"


def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _combinations(n: int, k: int) -> int:
    """C(n, k) — binomial coefficient."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _permutations(n: int, k: int) -> int:
    """P(n, k) = n! / (n-k)!"""
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticMathGenerator:
    """Generate synthetic math problems with exact, verifiable answers.

    Parameters
    ----------
    config:
        Generator configuration. Defaults to :class:`SyntheticMathConfig` with
        seed 42 when *None*.
    """

    def __init__(self, config: Optional[SyntheticMathConfig] = None) -> None:
        self.config = config if config is not None else SyntheticMathConfig()

    # ------------------------------------------------------------------
    # Problem-type generators
    # ------------------------------------------------------------------

    def _generate_arithmetic(self, difficulty: str, rng: random.Random) -> MathProblem:
        """Generate an arithmetic problem at the requested difficulty."""
        cap = max(10, self.config.max_number // 10)

        if difficulty == "easy":
            a = rng.randint(1, cap)
            b = rng.randint(1, cap)
            op = rng.choice(["+", "-", "*"])
            if op == "+":
                answer_val = a + b
                text = f"What is {a} + {b}?"
            elif op == "-":
                # keep answer non-negative for simplicity
                a, b = max(a, b), min(a, b)
                answer_val = a - b
                text = f"What is {a} - {b}?"
            else:
                answer_val = a * b
                text = f"What is {a} \u00d7 {b}?"
            answer = str(answer_val)
            answer_numeric: Optional[float] = float(answer_val)

        elif difficulty == "medium":
            a = rng.randint(1, cap)
            b = rng.randint(1, cap)
            c = rng.randint(1, cap)
            op1 = rng.choice(["+", "-"])
            answer_val = (a + b) * c if op1 == "+" else (a - b) * c
            text = f"What is ({a} {op1} {b}) \u00d7 {c}?"
            answer = str(answer_val)
            answer_numeric = float(answer_val)

        else:  # hard — fractions
            # a/b + c/d
            b = rng.randint(2, 12)
            d = rng.randint(2, 12)
            a = rng.randint(1, b * 3)
            c = rng.randint(1, d * 3)
            num = a * d + c * b
            den = b * d
            answer = _fraction_str(num, den)
            text = f"What is {a}/{b} + {c}/{d}? Express your answer as a reduced fraction."
            rn, rd = _reduce_fraction(num, den)
            answer_numeric = rn / rd

        return MathProblem(
            problem_id="",          # filled by generate()
            problem_text=text,
            answer=answer,
            answer_numeric=answer_numeric,
            difficulty=difficulty,
            problem_type="arithmetic",
        )

    def _generate_algebra(self, difficulty: str, rng: random.Random) -> MathProblem:
        """Generate an algebra problem at the requested difficulty."""
        if difficulty == "easy":
            # ax = b  →  x = b/a
            a = rng.randint(2, 20)
            x = rng.randint(1, 50)
            b = a * x
            text = f"Solve for x: {a}x = {b}"
            answer = str(x)
            answer_numeric: Optional[float] = float(x)

        elif difficulty == "medium":
            # ax + b = c  →  x = (c - b) / a
            a = rng.randint(2, 15)
            x = rng.randint(-20, 20)
            b = rng.randint(-30, 30)
            c = a * x + b
            sign = "+" if b >= 0 else "-"
            b_abs = abs(b)
            text = f"Solve for x: {a}x {sign} {b_abs} = {c}"
            answer = str(x)
            answer_numeric = float(x)

        else:  # hard — quadratic with integer roots
            # ax² + bx + c = 0, roots r1 and r2
            # a(x - r1)(x - r2) = 0
            a_coeff = rng.randint(1, 5)
            r1 = rng.randint(-10, 10)
            r2 = rng.randint(-10, 10)
            # expand: a*x^2 - a*(r1+r2)*x + a*r1*r2
            b_coeff = -a_coeff * (r1 + r2)
            c_coeff = a_coeff * r1 * r2
            # Format the equation
            def _fmt_coeff(coeff: int, var: str, first: bool) -> str:
                if coeff == 0:
                    return ""
                sign_str = "" if first else (" + " if coeff > 0 else " - ")
                abs_c = abs(coeff)
                if var == "":
                    return f"{sign_str}{abs_c}"
                prefix = "" if abs_c == 1 else str(abs_c)
                if first and coeff < 0:
                    return f"-{prefix}{var}"
                return f"{sign_str}{prefix}{var}"

            a_str = _fmt_coeff(a_coeff, "x\u00b2", True)
            b_str = _fmt_coeff(b_coeff, "x", False)
            c_str = _fmt_coeff(c_coeff, "", False)
            equation = f"{a_str}{b_str}{c_str} = 0"
            # Return roots sorted for determinism
            roots = sorted([r1, r2])
            answer = f"{roots[0]},{roots[1]}"
            text = f"Find the integer roots of: {equation} (give as x1,x2 with x1 \u2264 x2)"
            answer_numeric = None  # two values, no single float

        return MathProblem(
            problem_id="",
            problem_text=text,
            answer=answer,
            answer_numeric=answer_numeric,
            difficulty=difficulty,
            problem_type="algebra",
        )

    def _generate_combinatorics(self, difficulty: str, rng: random.Random) -> MathProblem:
        """Generate a combinatorics problem at the requested difficulty."""
        if difficulty == "easy":
            # n! for n in [3, 8]
            n = rng.randint(3, 8)
            answer_val = _factorial(n)
            text = f"How many ways can {n} distinct objects be arranged in a row? (Compute {n}!)"
            answer = str(answer_val)
            answer_numeric: Optional[float] = float(answer_val)

        elif difficulty == "medium":
            # C(n, k)
            n = rng.randint(5, 20)
            k = rng.randint(2, min(n - 1, 8))
            answer_val = _combinations(n, k)
            text = (
                f"In how many ways can you choose {k} items from a set of {n} distinct items? "
                f"(Compute C({n},{k}))"
            )
            answer = str(answer_val)
            answer_numeric = float(answer_val)

        else:  # hard — permutations with a restriction
            # P(n, k) = n! / (n-k)! ordered selections
            n = rng.randint(6, 15)
            k = rng.randint(3, min(n, 6))
            answer_val = _permutations(n, k)
            text = (
                f"How many ordered sequences of {k} distinct elements can be chosen "
                f"from {n} distinct elements? (Compute P({n},{k}))"
            )
            answer = str(answer_val)
            answer_numeric = float(answer_val)

        return MathProblem(
            problem_id="",
            problem_text=text,
            answer=answer,
            answer_numeric=answer_numeric,
            difficulty=difficulty,
            problem_type="combinatorics",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int,
        problem_type: Optional[str] = None,
    ) -> list[MathProblem]:
        """Generate *n* synthetic math problems.

        Parameters
        ----------
        n:
            Number of problems to generate.
        problem_type:
            If given, all problems will be of this type. Otherwise the type is
            sampled uniformly from :attr:`config.problem_types`.

        Returns
        -------
        list[MathProblem]
            Problems with unique ``problem_id`` strings.
        """
        cfg = self.config
        rng = random.Random(cfg.seed)

        difficulties = list(cfg.difficulty_distribution.keys())
        weights = [cfg.difficulty_distribution[d] for d in difficulties]

        _type_generators = {
            "arithmetic": self._generate_arithmetic,
            "algebra": self._generate_algebra,
            "combinatorics": self._generate_combinatorics,
        }

        problems: list[MathProblem] = []
        for i in range(n):
            diff = rng.choices(difficulties, weights=weights, k=1)[0]
            ptype = problem_type if problem_type is not None else rng.choice(cfg.problem_types)

            gen_fn = _type_generators[ptype]
            problem = gen_fn(diff, rng)
            problem.problem_id = f"{ptype}_{diff}_{i:05d}"
            problems.append(problem)

        return problems

    def verify(self, problem: MathProblem, candidate_answer: str) -> bool:
        """Return *True* if *candidate_answer* matches *problem.answer*.

        Comparison is performed:

        1. As normalised strings (stripped whitespace, lower-case).
        2. As floats when both sides are numeric.
        """
        canonical = problem.answer.strip().lower()
        candidate = candidate_answer.strip().lower()

        if canonical == candidate:
            return True

        # Numeric comparison
        try:
            cand_f = float(candidate)
            # Try parsing the canonical answer as float too
            try:
                can_f = float(canonical)
            except ValueError:
                # canonical might be a fraction
                if "/" in canonical:
                    num_s, den_s = canonical.split("/", 1)
                    can_f = float(num_s) / float(den_s)
                else:
                    return False
            return abs(cand_f - can_f) < 1e-9
        except (ValueError, ZeroDivisionError):
            return False

    def difficulty_stats(self, problems: list[MathProblem]) -> dict:
        """Return summary statistics for a list of problems.

        Returns
        -------
        dict with keys:
            ``total``         — total number of problems
            ``by_difficulty`` — mapping difficulty → count
            ``by_type``       — mapping problem_type → count
        """
        by_diff: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for p in problems:
            by_diff[p.difficulty] = by_diff.get(p.difficulty, 0) + 1
            by_type[p.problem_type] = by_type.get(p.problem_type, 0) + 1
        return {
            "total": len(problems),
            "by_difficulty": by_diff,
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Registry hook (wired in src/data/__init__.py)
# ---------------------------------------------------------------------------

DATA_REGISTRY: dict[str, type] = {
    "synthetic_math": SyntheticMathGenerator,
}
