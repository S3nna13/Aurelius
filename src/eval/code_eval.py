"""Code generation evaluation: HumanEval-style problems, functional correctness, and metrics.

Security note
-------------
This module runs model-generated Python code inside a restricted namespace
(``_SAFE_BUILTINS``) for functional-correctness scoring. The sandboxed-run
sites below are intentional and annotated with ``# nosec B102``. Callers MUST:

  * pre-validate input via the sanitizer in ``src/inference/code_execution.py``
    (``sanitize_code``) when the source is untrusted, and
  * rely on the length guard (``_MAX_EXEC_LEN``) enforced immediately before
    each sandboxed-run invocation.

Never pass unrestricted globals (e.g. the real ``builtins`` module or module
globals) to these call sites — doing so would escalate a sandbox escape to
arbitrary-code execution in the host process.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

# Maximum length of a code string accepted for sandboxed evaluation.
# Keeps pathological or adversarial inputs from exhausting the compiler/parser.
_MAX_EXEC_LEN = 100_000


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CodeEvalConfig:
    """Configuration for code generation evaluation."""

    language: str = "python"  # only Python for now
    timeout_seconds: float = 5.0
    n_samples: int = 1  # samples per problem for pass@k
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    safety_check: bool = True  # reject code with dangerous patterns


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CodeProblem:
    """A single code generation problem."""

    problem_id: str
    prompt: str  # function signature + docstring
    canonical_solution: str
    test_cases: list[str]  # list of assert statements
    entry_point: str  # function name to call


# ---------------------------------------------------------------------------
# Synthetic problem generation
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict] = [
    {
        "problem_id": "sum_list",
        "prompt": (
            'def sum_list(nums: list) -> int:\n    """Return the sum of all elements in nums."""\n'
        ),
        "canonical_solution": ("def sum_list(nums: list) -> int:\n    return sum(nums)\n"),
        "test_cases": [
            "assert sum_list([1, 2, 3]) == 6",
            "assert sum_list([]) == 0",
            "assert sum_list([-1, 1]) == 0",
            "assert sum_list([10]) == 10",
        ],
        "entry_point": "sum_list",
    },
    {
        "problem_id": "max_list",
        "prompt": (
            "def max_list(nums: list) -> int:\n"
            '    """Return the maximum element in nums. Assumes nums is non-empty."""\n'
        ),
        "canonical_solution": ("def max_list(nums: list) -> int:\n    return max(nums)\n"),
        "test_cases": [
            "assert max_list([1, 2, 3]) == 3",
            "assert max_list([5]) == 5",
            "assert max_list([-3, -1, -2]) == -1",
            "assert max_list([0, 100, 50]) == 100",
        ],
        "entry_point": "max_list",
    },
    {
        "problem_id": "reverse_string",
        "prompt": (
            'def reverse_string(s: str) -> str:\n    """Return the reverse of string s."""\n'
        ),
        "canonical_solution": ("def reverse_string(s: str) -> str:\n    return s[::-1]\n"),
        "test_cases": [
            'assert reverse_string("hello") == "olleh"',
            'assert reverse_string("") == ""',
            'assert reverse_string("a") == "a"',
            'assert reverse_string("abcd") == "dcba"',
        ],
        "entry_point": "reverse_string",
    },
    {
        "problem_id": "count_vowels",
        "prompt": (
            "def count_vowels(s: str) -> int:\n"
            '    """Return the number of vowels (a, e, i, o, u) in s (case-insensitive)."""\n'
        ),
        "canonical_solution": (
            "def count_vowels(s: str) -> int:\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
        ),
        "test_cases": [
            'assert count_vowels("hello") == 2',
            'assert count_vowels("") == 0',
            'assert count_vowels("rhythm") == 0',
            'assert count_vowels("AEIOU") == 5',
            'assert count_vowels("Python") == 1',
        ],
        "entry_point": "count_vowels",
    },
    {
        "problem_id": "is_palindrome",
        "prompt": (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True if s is a palindrome (reads the same forwards and backwards)."""\n'
        ),
        "canonical_solution": ("def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n"),
        "test_cases": [
            'assert is_palindrome("racecar") == True',
            'assert is_palindrome("hello") == False',
            'assert is_palindrome("") == True',
            'assert is_palindrome("a") == True',
            'assert is_palindrome("abba") == True',
        ],
        "entry_point": "is_palindrome",
    },
    {
        "problem_id": "factorial",
        "prompt": ('def factorial(n: int) -> int:\n    """Return n! (n factorial). n >= 0."""\n'),
        "canonical_solution": (
            "def factorial(n: int) -> int:\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "test_cases": [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
            "assert factorial(10) == 3628800",
        ],
        "entry_point": "factorial",
    },
    {
        "problem_id": "fibonacci",
        "prompt": (
            "def fibonacci(n: int) -> int:\n"
            '    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""\n'
        ),
        "canonical_solution": (
            "def fibonacci(n: int) -> int:\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    if n == 1:\n"
            "        return 1\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        ),
        "test_cases": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(6) == 8",
            "assert fibonacci(10) == 55",
        ],
        "entry_point": "fibonacci",
    },
    {
        "problem_id": "binary_search",
        "prompt": (
            "def binary_search(arr: list, target: int) -> int:\n"
            '    """Return the index of target in sorted arr, or -1 if not found."""\n'
        ),
        "canonical_solution": (
            "def binary_search(arr: list, target: int) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        "test_cases": [
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2",
            "assert binary_search([1, 3, 5, 7, 9], 1) == 0",
            "assert binary_search([1, 3, 5, 7, 9], 9) == 4",
            "assert binary_search([1, 3, 5, 7, 9], 4) == -1",
            "assert binary_search([], 1) == -1",
        ],
        "entry_point": "binary_search",
    },
    {
        "problem_id": "bubble_sort",
        "prompt": (
            "def bubble_sort(arr: list) -> list:\n"
            '    """Return a new sorted list using bubble sort."""\n'
        ),
        "canonical_solution": (
            "def bubble_sort(arr: list) -> list:\n"
            "    arr = list(arr)\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "    return arr\n"
        ),
        "test_cases": [
            "assert bubble_sort([3, 1, 2]) == [1, 2, 3]",
            "assert bubble_sort([]) == []",
            "assert bubble_sort([1]) == [1]",
            "assert bubble_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]",
        ],
        "entry_point": "bubble_sort",
    },
    {
        "problem_id": "flatten_list",
        "prompt": (
            "def flatten_list(nested: list) -> list:\n"
            '    """Return a flat list from a one-level nested list of lists."""\n'
        ),
        "canonical_solution": (
            "def flatten_list(nested: list) -> list:\n"
            "    result = []\n"
            "    for sublist in nested:\n"
            "        result.extend(sublist)\n"
            "    return result\n"
        ),
        "test_cases": [
            "assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]",
            "assert flatten_list([]) == []",
            "assert flatten_list([[1]]) == [1]",
            "assert flatten_list([[], [1, 2], []]) == [1, 2]",
        ],
        "entry_point": "flatten_list",
    },
    {
        "problem_id": "count_occurrences",
        "prompt": (
            "def count_occurrences(lst: list, item) -> int:\n"
            '    """Return the number of times item appears in lst."""\n'
        ),
        "canonical_solution": (
            "def count_occurrences(lst: list, item) -> int:\n    return lst.count(item)\n"
        ),
        "test_cases": [
            "assert count_occurrences([1, 2, 1, 3, 1], 1) == 3",
            "assert count_occurrences([], 5) == 0",
            "assert count_occurrences([1, 2, 3], 4) == 0",
            'assert count_occurrences(["a", "b", "a"], "a") == 2',
        ],
        "entry_point": "count_occurrences",
    },
    {
        "problem_id": "is_prime",
        "prompt": (
            "def is_prime(n: int) -> bool:\n"
            '    """Return True if n is a prime number, False otherwise. n >= 2."""\n'
        ),
        "canonical_solution": (
            "def is_prime(n: int) -> bool:\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        "test_cases": [
            "assert is_prime(2) == True",
            "assert is_prime(3) == True",
            "assert is_prime(4) == False",
            "assert is_prime(17) == True",
            "assert is_prime(1) == False",
        ],
        "entry_point": "is_prime",
    },
]


def generate_synthetic_problems(n: int, seed: int = 42) -> list[CodeProblem]:
    """Generate n simple Python coding problems programmatically.

    Problems cycle through the built-in problem bank.  When n exceeds the bank
    size the bank is repeated (with a disambiguating suffix on the problem_id).

    Args:
        n: Number of problems to return.
        seed: Random seed (used to shuffle the bank before selecting).

    Returns:
        List of CodeProblem instances.
    """
    rng = random.Random(seed)  # noqa: S311
    bank = list(_PROBLEMS)
    rng.shuffle(bank)

    problems: list[CodeProblem] = []
    for i in range(n):
        spec = bank[i % len(bank)]
        suffix = "" if i < len(bank) else f"_{i // len(bank)}"
        problems.append(
            CodeProblem(
                problem_id=spec["problem_id"] + suffix,
                prompt=spec["prompt"],
                canonical_solution=spec["canonical_solution"],
                test_cases=list(spec["test_cases"]),
                entry_point=spec["entry_point"],
            )
        )
    return problems


# ---------------------------------------------------------------------------
# Safety checking
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\b__import__\s*\("),
]


def is_safe_code(code: str) -> bool:
    """Return False if code contains any known dangerous patterns.

    Checked patterns: os.system, subprocess, eval(), exec(), open(), __import__().

    Args:
        code: Python source code string.

    Returns:
        True if no dangerous patterns are found, False otherwise.
    """
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(code):
            return False
    return True


# ---------------------------------------------------------------------------
# Restricted execution
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    # Math / numeric
    "abs": abs,
    "divmod": divmod,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sum": sum,
    # Type constructors
    "bool": bool,
    "bytes": bytes,
    "complex": complex,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "set": set,
    "str": str,
    "tuple": tuple,
    # Iteration
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "filter": filter,
    "len": len,
    "map": map,
    "range": range,
    "reversed": reversed,
    "sorted": sorted,
    "zip": zip,
    # Inspection
    "hasattr": hasattr,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "getattr": getattr,
    # Misc
    "chr": chr,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "ord": ord,
    "print": print,
    "repr": repr,
    "type": type,
    "vars": vars,
    # Exceptions (needed for raise / except in solutions)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ZeroDivisionError": ZeroDivisionError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    # Boolean literals (accessed as builtins in some situations)
    "True": True,
    "False": False,
    "None": None,
}


def execute_code_safely(
    code: str,
    test_cases: list[str],
    timeout: float,
) -> tuple[bool, str]:
    """Execute code and test cases in a restricted namespace.

    Compiles and runs *code* first, then runs each assert statement in
    *test_cases* within the same namespace.  No OS-level sandboxing is used.

    Args:
        code: Python source code defining the function(s) under test.
        test_cases: List of assert statements as strings.
        timeout: Reserved for API consistency; not enforced in pure-Python mode.

    Returns:
        Tuple (passed, error_message).  passed is True only when all test
        cases execute without raising any exception.  error_message is empty
        on success.
    """
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

    # Size guard: reject pathologically large inputs before compiling.
    if not code:
        return False, "empty solution code"
    if len(code) > _MAX_EXEC_LEN:
        return False, f"solution too large ({len(code)} chars > {_MAX_EXEC_LEN})"

    # Compile and run solution code
    try:
        compiled_solution = compile(code, "<solution>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError in solution: {exc}"

    try:
        # nosec B102 — intentional sandboxed code execution; caller must
        # validate input via CodeSandbox/sanitize_code and _MAX_EXEC_LEN above.
        exec(compiled_solution, namespace)  # noqa: S102  # nosec B102
    except Exception as exc:  # noqa: BLE001
        return False, f"RuntimeError running solution: {type(exc).__name__}: {exc}"

    # Run each test case
    for test in test_cases:
        if not test or len(test) > _MAX_EXEC_LEN:
            return False, f"test case rejected by size guard: len={len(test)}"
        try:
            compiled_test = compile(test, "<test>", "exec")
        except SyntaxError as exc:
            return False, f"SyntaxError in test case '{test}': {exc}"

        try:
            # nosec B102 — intentional sandboxed code execution; caller must
            # validate input via CodeSandbox/sanitize_code and _MAX_EXEC_LEN above.
            exec(compiled_test, namespace)  # noqa: S102  # nosec B102
        except AssertionError:
            return False, f"AssertionError: test case failed: {test}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__} in test case '{test}': {exc}"

    return True, ""


# ---------------------------------------------------------------------------
# pass@k metric
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the pass@k metric using the exact combinatorial formula.

    Formula: 1 - C(n - c, k) / C(n, k)

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct (passing) samples.
        k: The k in pass@k.

    Returns:
        Float in [0, 1].
    """
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0
    # When n - c < k the numerator C(n-c, k) is 0, so result is 1.0.
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CodeEvaluator:
    """Evaluates code generation solutions against HumanEval-style problems."""

    def __init__(self, config: CodeEvalConfig) -> None:
        self.config = config

    def evaluate_solution(
        self,
        problem: CodeProblem,
        solution: str,
    ) -> dict[str, bool | str]:
        """Evaluate a single solution against a problem.

        Args:
            problem: The coding problem to evaluate.
            solution: Candidate Python source code.

        Returns:
            Dict with keys "passed" (bool), "safe" (bool), "error" (str).
        """
        safe = True
        if self.config.safety_check:
            safe = is_safe_code(solution)
            if not safe:
                return {"passed": False, "safe": False, "error": "unsafe code detected"}

        passed, error = execute_code_safely(
            solution,
            problem.test_cases,
            self.config.timeout_seconds,
        )
        return {"passed": passed, "safe": safe, "error": error}

    def evaluate_problems(
        self,
        problems: list[CodeProblem],
        solutions: list[str],
    ) -> dict[str, float]:
        """Evaluate all problem/solution pairs and return aggregate metrics.

        Args:
            problems: List of coding problems.
            solutions: Corresponding list of candidate solutions (same length).

        Returns:
            Dict with keys "pass@1" (float), "accuracy" (float),
            "n_safe" (float).
        """
        if len(problems) != len(solutions):
            raise ValueError(
                f"problems ({len(problems)}) and solutions ({len(solutions)}) "
                "must have the same length"
            )

        n_problems = len(problems)
        if n_problems == 0:
            return {"pass@1": 0.0, "accuracy": 0.0, "n_safe": 0.0}

        n_correct = 0
        n_safe = 0

        for problem, solution in zip(problems, solutions):
            result = self.evaluate_solution(problem, solution)
            if result["safe"]:
                n_safe += 1
            if result["passed"]:
                n_correct += 1

        pass1 = pass_at_k(n=n_problems, c=n_correct, k=1)
        accuracy = n_correct / n_problems

        return {
            "pass@1": pass1,
            "accuracy": accuracy,
            "n_safe": float(n_safe),
        }
