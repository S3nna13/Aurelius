"""Synthetic code data generation pipeline for Aurelius.

Inspired by OSS-Instruct (Magicoder), Evol-Instruct for code, and
StarCoder's fill-in-the-middle approach. Generates instruction-following
training data from seed code snippets using templates, mutations, and a
pluggable generate_fn.

Pure Python — no external APIs, no HuggingFace, no PyTorch required.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CodeExample:
    instruction: str
    code: str
    language: str = "python"
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    tags: list[str] = field(default_factory=list)
    test_cases: list[str] = field(default_factory=list)
    source: str = "synthetic"


@dataclass
class MutationConfig:
    rename_variables: bool = True
    add_comments: bool = True
    change_data_structures: bool = False
    add_type_hints: bool = True
    wrap_in_class: bool = False


# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

_BRANCH_PATTERN = re.compile(r"\b(if|elif|for|while|try|except|and|or)\b")


def count_cyclomatic_complexity(code: str) -> int:
    """Approximation: 1 + count of branching keywords."""
    return 1 + len(_BRANCH_PATTERN.findall(code))


def classify_difficulty(code: str) -> str:
    """'easy' (complexity ≤ 3), 'medium' (4-7), 'hard' (≥ 8)."""
    c = count_cyclomatic_complexity(code)
    if c <= 3:
        return "easy"
    if c <= 7:
        return "medium"
    return "hard"


def extract_function_names(code: str) -> list[str]:
    """Parse Python code with ast; return list of function def names."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def extract_imports(code: str) -> list[str]:
    """Return list of imported module names."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def rename_variables(code: str, mapping: dict[str, str] | None = None) -> str:
    """Rename local variables using a mapping or auto-generated names."""
    if mapping is None:
        # Extract simple identifiers not on def/class/import lines
        candidates = re.findall(r"\b([a-z][a-z0-9_]*)\b", code)
        # Exclude Python keywords and common builtins
        _keywords = {
            "if",
            "else",
            "elif",
            "for",
            "while",
            "in",
            "not",
            "and",
            "or",
            "return",
            "import",
            "from",
            "class",
            "def",
            "pass",
            "break",
            "continue",
            "try",
            "except",
            "finally",
            "with",
            "as",
            "True",
            "False",
            "None",
            "lambda",
            "yield",
            "raise",
            "global",
            "nonlocal",
            "del",
            "assert",
            "print",
            "len",
            "range",
            "int",
            "str",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "sum",
            "min",
            "max",
            "self",
            "cls",
            "args",
            "kwargs",
        }
        unique = [c for c in dict.fromkeys(candidates) if c not in _keywords]
        letters = "abcdefghijklmnopqrstuvwxyz"
        mapping = {
            var: letters[i % len(letters)] + (str(i // len(letters)) if i >= len(letters) else "")
            for i, var in enumerate(unique)
        }

    result = code
    for old, new in mapping.items():
        result = re.sub(rf"\b{re.escape(old)}\b", new, result)
    return result


def add_type_hints(code: str) -> str:
    """Add '-> None' to function signatures lacking a return annotation."""

    def _annotate(match: re.Match) -> str:
        sig = match.group(0)
        # Already has return annotation
        if "->" in sig:
            return sig
        # Insert '-> None' before the colon
        return sig.rstrip(":") + " -> None:"

    pattern = re.compile(r"def\s+\w+\s*\([^)]*\)\s*:")
    return pattern.sub(_annotate, code)


def add_docstring(
    code: str,
    docstring: str = "Process the input and return the result.",
) -> str:
    """Insert a docstring after the first 'def ...:' line if none exists."""
    lines = code.split("\n")
    result: list[str] = []
    inserted = False
    i = 0
    while i < len(lines):
        result.append(lines[i])
        if not inserted and re.match(r"\s*def\s+\w+.*:", lines[i]):
            # Check if next non-blank line is already a docstring
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and not (
                lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''")
            ):
                # Detect indentation
                m = re.match(r"(\s*)", lines[i])
                indent = (m.group(1) if m else "") + "    "
                result.append(f'{indent}"""{docstring}"""')
                inserted = True
        i += 1
    return "\n".join(result)


def apply_mutations(code: str, config: MutationConfig) -> str:
    """Apply all enabled mutations from config."""
    result = code
    if config.rename_variables:
        result = rename_variables(result)
    if config.add_type_hints:
        result = add_type_hints(result)
    if config.add_comments:
        # Add a simple header comment if not present
        if not result.lstrip().startswith("#"):
            result = "# Auto-generated code\n" + result
    return result


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------


def build_oss_instruct_prompt(seed_code: str, language: str = "python") -> str:
    """Build OSS-Instruct style prompt from seed code."""
    return (
        f"Here is a {language} code snippet:\n\n"
        f"```{language}\n{seed_code}\n```\n\n"
        "Based on the concepts and patterns in the code above, "
        "write a programming problem that would require implementing "
        "similar logic. The problem should be self-contained and clear.\n\n"
        "Problem:"
    )


def build_instruction_from_code(
    code: str,
    generate_fn: Callable[[str], str],
    language: str = "python",
) -> str:
    """Build a description of what the code does using generate_fn."""
    prompt = (
        f"Describe what the following {language} code does in one sentence "
        "as an instruction for a programmer:\n\n"
        f"```{language}\n{code}\n```\n\nInstruction:"
    )
    return generate_fn(prompt)


# ---------------------------------------------------------------------------
# Test case generation
# ---------------------------------------------------------------------------


def parse_test_cases_from_text(text: str) -> list[str]:
    """Parse test function definitions from raw text."""
    lines = text.split("\n")
    cases: list[str] = []
    current: list[str] = []
    in_test = False

    for line in lines:
        if re.match(r"\s*def\s+test_\w+", line):
            if current:
                cases.append("\n".join(current))
            current = [line]
            in_test = True
        elif in_test:
            if line.strip() == "" and current:
                # blank line ends the test block
                cases.append("\n".join(current))
                current = []
                in_test = False
            else:
                current.append(line)

    if current:
        cases.append("\n".join(current))

    return [c for c in cases if c.strip()]


def generate_test_cases(
    function_name: str,
    generate_fn: Callable[[str], str],
    n_cases: int = 3,
) -> list[str]:
    """Use generate_fn to produce unit test cases for function_name."""
    prompt = (
        f"Write {n_cases} pytest unit test functions for a function called "
        f"'{function_name}'. Include edge cases.\n\n"
        "Tests:"
    )
    raw = generate_fn(prompt)
    return parse_test_cases_from_text(raw)


# ---------------------------------------------------------------------------
# Seed snippets
# ---------------------------------------------------------------------------

SEED_SNIPPETS: list[str] = [
    "def add(a, b):\n    return a + b\n",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1\n",  # noqa: E501
    "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n",  # noqa: E501
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop()\n",  # noqa: E501
    "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",  # noqa: E501
    "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n",  # noqa: E501
    "def count_words(text):\n    words = text.split()\n    freq = {}\n    for w in words:\n        freq[w] = freq.get(w, 0) + 1\n    return freq\n",  # noqa: E501
    "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    cols_B = len(B[0])\n    C = [[0]*cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                C[i][j] += A[i][k] * B[k][j]\n    return C\n",  # noqa: E501
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",  # noqa: E501
    "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        if target - num in seen:\n            return [seen[target-num], i]\n        seen[num] = i\n    return []\n",  # noqa: E501
    "def longest_common_prefix(strs):\n    if not strs: return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n        if not prefix: return ''\n    return prefix\n",  # noqa: E501
    "def rotate_matrix(matrix):\n    n = len(matrix)\n    for i in range(n):\n        for j in range(i, n):\n            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n    for row in matrix:\n        row.reverse()\n    return matrix\n",  # noqa: E501
    "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            stack.append(char)\n    return not stack\n",  # noqa: E501
    "def roman_to_int(s):\n    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    total = 0\n    for i in range(len(s)):\n        if i+1 < len(s) and vals[s[i]] < vals[s[i+1]]:\n            total -= vals[s[i]]\n        else:\n            total += vals[s[i]]\n    return total\n",  # noqa: E501
    "def power(base, exp):\n    if exp == 0: return 1\n    if exp % 2 == 0:\n        half = power(base, exp // 2)\n        return half * half\n    return base * power(base, exp - 1)\n",  # noqa: E501
    "def compress_string(s):\n    if not s: return s\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(s[i-1] + (str(count) if count > 1 else ''))\n            count = 1\n    result.append(s[-1] + (str(count) if count > 1 else ''))\n    return ''.join(result)\n",  # noqa: E501
    "def find_missing_number(nums):\n    n = len(nums)\n    expected = n * (n + 1) // 2\n    return expected - sum(nums)\n",  # noqa: E501
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SyntheticCodePipeline:
    """End-to-end synthetic code data generation pipeline."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        seed_snippets: list[str] | None = None,
        languages: list[str] | None = None,
        mutation_config: MutationConfig | None = None,
        max_examples: int = 100,
    ) -> None:
        self.generate_fn = generate_fn
        self.seed_snippets = seed_snippets if seed_snippets is not None else SEED_SNIPPETS
        self.languages = languages if languages is not None else ["python"]
        self.mutation_config = mutation_config if mutation_config is not None else MutationConfig()
        self.max_examples = max_examples
        self._generated: list[CodeExample] = []

    def generate_from_seed(self, seed_code: str) -> CodeExample:
        """Generate one CodeExample from a seed snippet."""
        lang = self.languages[0]

        # Build OSS-Instruct prompt and call generate_fn for the instruction
        prompt = build_oss_instruct_prompt(seed_code, lang)
        instruction = self.generate_fn(prompt).strip() or "Implement the function."

        # Mutate the seed code
        mutated_code = apply_mutations(seed_code, self.mutation_config)

        # Classify difficulty
        difficulty = classify_difficulty(seed_code)

        # Extract tags from original code
        fn_names = extract_function_names(seed_code)
        imports = extract_imports(seed_code)
        tags = fn_names + imports

        return CodeExample(
            instruction=instruction,
            code=mutated_code,
            language=lang,
            difficulty=difficulty,
            tags=tags,
            source="synthetic",
        )

    def run(self, n_examples: int = 10) -> list[CodeExample]:
        """Generate n_examples by cycling through seed_snippets."""
        generated: list[CodeExample] = []
        seeds = self.seed_snippets
        for i in range(min(n_examples, self.max_examples)):
            seed = seeds[i % len(seeds)]
            try:
                ex = self.generate_from_seed(seed)
                generated.append(ex)
            except Exception:  # noqa: S110
                pass
        self._generated = generated
        return generated

    def export(self, examples: list[CodeExample]) -> list[dict]:
        """Export as list of dicts."""
        return [
            {
                "instruction": ex.instruction,
                "code": ex.code,
                "difficulty": ex.difficulty,
                "language": ex.language,
                "tags": ex.tags,
            }
            for ex in examples
        ]

    def deduplicate(self, examples: list[CodeExample]) -> list[CodeExample]:
        """Remove exact duplicates by md5 hash of code string."""
        seen: set = set()
        result: list[CodeExample] = []
        for ex in examples:
            h = hashlib.md5(ex.code.encode(), usedforsecurity=False).hexdigest()
            if h not in seen:
                seen.add(h)
                result.append(ex)
        return result
