"""Self-Instruct (Wang et al. 2022) synthetic data generation pipeline.

Generates instruction-following training data using a pluggable generate() function,
with ROUGE-L deduplication, quality filtering, and classification detection.
No external APIs required — framework is fully self-contained.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Instruction:
    """A single instruction-following example."""

    instruction: str
    input: str = ""
    output: str = ""
    is_classification: bool = False
    source: str = "generated"  # "seed" or "generated"


# ---------------------------------------------------------------------------
# ROUGE-L implementation (pure Python)
# ---------------------------------------------------------------------------


def lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence of two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use two-row DP to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer (lowercased)."""
    return text.lower().split()


def rouge_l_similarity(a: str, b: str) -> float:
    """Compute ROUGE-L F1 between two strings using pure Python LCS.

    Returns a float in [0.0, 1.0].  Returns 0.0 for empty inputs.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        # Both empty → treat as identical (1.0); one empty → 0.0
        if not tokens_a and not tokens_b:
            return 1.0
        return 0.0
    lcs = lcs_length(tokens_a, tokens_b)
    precision = lcs / len(tokens_b)
    recall = lcs / len(tokens_a)
    if precision + recall == 0.0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def filter_instruction(
    instruction: str,
    existing_instructions: list[str],
    rouge_threshold: float = 0.7,
    min_length: int = 10,
    max_length: int = 500,
) -> bool:
    """Return True if instruction passes all quality filters.

    Filters applied:
    1. Length must be between min_length and max_length (inclusive).
    2. ROUGE-L similarity vs every existing instruction must be < rouge_threshold.
    """
    if len(instruction) < min_length or len(instruction) > max_length:
        return False
    for existing in existing_instructions:
        if rouge_l_similarity(instruction, existing) >= rouge_threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Classification heuristic
# ---------------------------------------------------------------------------

_CLASSIFICATION_KEYWORDS = [
    r"\bis\b.*\bor\b",
    r"\bclassify\b",
    r"\bcategorize\b",
    r"\bwhich (category|type|class|label)\b",
    r"\bdetect\b",
    r"\bidentify (whether|if)\b",
    r"\bpositive or negative\b",
    r"\btrue or false\b",
    r"\byes or no\b",
    r"\bdetermine (if|whether)\b",
    r"\blabel\b.*\b(as|into)\b",
]

_CLASSIFICATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CLASSIFICATION_KEYWORDS]


def is_classification_task(instruction: str) -> bool:
    """Heuristic classifier: does this instruction ask for classification/categorization?"""
    for pattern in _CLASSIFICATION_PATTERNS:
        if pattern.search(instruction):
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_instruction_prompt(sampled: list[Instruction], n_examples: int = 8) -> str:
    """Build the few-shot prompt for instruction generation.

    Uses up to n_examples instructions from `sampled` as demonstrations,
    then asks the model to generate a new instruction.
    """
    examples = sampled[:n_examples]
    lines = [
        "Come up with a series of tasks and instructions for an AI assistant to follow.",
        "Here are some examples of tasks:\n",
    ]
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"{idx}. {ex.instruction}")
    lines.append(f"\n{len(examples) + 1}.")
    return "\n".join(lines)


def build_instance_prompt(instruction: Instruction) -> str:
    """Build the prompt for generating input/output instances for an instruction."""
    lines = [
        "Given the following task, generate an appropriate input (if required) and output.",
        "",
        f"Task: {instruction.instruction}",
        "",
    ]
    if instruction.is_classification:
        lines += [
            "This is a classification task.  Generate a realistic input and the correct label.",
            "",
        ]
    else:
        lines += [
            "Generate a concrete input (if the task requires one) and a high-quality output.",
            "",
        ]
    lines += [
        "Input:",
        "",
        "Output:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_generated_instructions(raw_text: str) -> list[str]:
    """Parse model output into individual instruction strings.

    Handles numbered lists like:
        1. Do something.
        2. Write a poem about X.
    and plain newline-separated text.
    """
    instructions: list[str] = []
    # Try to match numbered items first (e.g. "1. ...", "1) ...")
    numbered = re.findall(r"^\s*\d+[.)]\s+(.+)", raw_text, re.MULTILINE)
    if numbered:
        for item in numbered:
            item = item.strip()
            if item:
                instructions.append(item)
    else:
        # Fall back: split on newlines
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                instructions.append(line)
    return instructions


# ---------------------------------------------------------------------------
# Instruction pool
# ---------------------------------------------------------------------------


class InstructionPool:
    """Manages seed + generated instructions with ROUGE-L deduplication."""

    def __init__(
        self,
        seed_instructions: list[Instruction],
        rouge_threshold: float = 0.7,
        seed: int = 42,
    ) -> None:
        self._pool: list[Instruction] = list(seed_instructions)
        self._rouge_threshold = rouge_threshold
        self._rng = random.Random(seed)

    def add(self, instr: Instruction) -> bool:
        """Add instruction if ROUGE-L similarity < threshold vs all existing.

        Returns True if the instruction was added, False if it was rejected.
        """
        existing_texts = [i.instruction for i in self._pool]
        if not filter_instruction(
            instr.instruction,
            existing_texts,
            rouge_threshold=self._rouge_threshold,
        ):
            return False
        self._pool.append(instr)
        return True

    def sample(self, n: int = 8) -> list[Instruction]:
        """Sample n instructions for few-shot prompt (mix of seed + generated)."""
        n = min(n, len(self._pool))
        return self._rng.sample(self._pool, n)

    def __len__(self) -> int:
        return len(self._pool)

    def to_list(self) -> list[Instruction]:
        """Return a copy of all instructions in the pool."""
        return list(self._pool)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SelfInstructPipeline:
    """Full Self-Instruct pipeline (Wang et al. 2022).

    Uses a pluggable generate_fn (prompt → text) so the same framework
    works with any language model backend or mock function.
    """

    def __init__(
        self,
        seed_instructions: list[Instruction],
        generate_fn: Callable[[str], str],
        rouge_threshold: float = 0.7,
        n_few_shot: int = 8,
        max_instructions: int = 100,
    ) -> None:
        self._generate_fn = generate_fn
        self._rouge_threshold = rouge_threshold
        self._n_few_shot = n_few_shot
        self._max_instructions = max_instructions
        self._generated: list[Instruction] = []
        self._pool = InstructionPool(
            seed_instructions,
            rouge_threshold=rouge_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(self, n_new: int = 20) -> list[Instruction]:
        """Run one cycle: sample, prompt, generate, filter.

        Returns the list of new Instruction objects that were successfully
        added to the pool in this cycle.
        """
        added: list[Instruction] = []

        if len(self._pool) >= self._max_instructions:
            return added

        # Build prompt from a sample of existing instructions
        sampled = self._pool.sample(self._n_few_shot)
        prompt = build_instruction_prompt(sampled, n_examples=self._n_few_shot)

        # Call the (potentially expensive) generate function once per batch
        raw_output = self._generate_fn(prompt)
        candidates = parse_generated_instructions(raw_output)

        existing_texts = [i.instruction for i in self._pool.to_list()]

        for candidate_text in candidates:
            if len(added) >= n_new:
                break
            if len(self._pool) >= self._max_instructions:
                break

            # Quality filter
            if not filter_instruction(
                candidate_text,
                existing_texts,
                rouge_threshold=self._rouge_threshold,
            ):
                continue

            # Classify task type
            is_cls = is_classification_task(candidate_text)

            # Optionally generate instance (input/output) — use generate_fn
            new_instr = Instruction(
                instruction=candidate_text,
                input="",
                output="",
                is_classification=is_cls,
                source="generated",
            )

            # Generate instance for the instruction
            instance_prompt = build_instance_prompt(new_instr)
            instance_raw = self._generate_fn(instance_prompt)
            # Parse a simple "Input: ... Output: ..." response
            in_match = re.search(r"Input:\s*(.+?)(?=Output:|$)", instance_raw, re.DOTALL)
            out_match = re.search(r"Output:\s*(.+)", instance_raw, re.DOTALL)
            new_instr.input = in_match.group(1).strip() if in_match else ""
            new_instr.output = out_match.group(1).strip() if out_match else instance_raw.strip()

            # Filter low-quality outputs
            if _is_low_quality_output(new_instr):
                continue

            if self._pool.add(new_instr):
                existing_texts.append(candidate_text)
                self._generated.append(new_instr)
                added.append(new_instr)

        return added

    def run(self, n_iterations: int = 5, n_per_iter: int = 10) -> list[Instruction]:
        """Run full self-instruct loop.

        Returns all generated (non-seed) instructions accumulated so far.
        """
        for _ in range(n_iterations):
            if len(self._pool) >= self._max_instructions:
                break
            self.generate_batch(n_new=n_per_iter)
        return list(self._generated)

    def export_sft_dataset(self) -> list[dict[str, str]]:
        """Export as list of {"instruction": ..., "input": ..., "output": ...} dicts."""
        results = []
        for instr in self._pool.to_list():
            results.append(
                {
                    "instruction": instr.instruction,
                    "input": instr.input,
                    "output": instr.output,
                }
            )
        return results

    @property
    def pool_size(self) -> int:
        """Total instructions in pool (seed + generated)."""
        return len(self._pool)


# ---------------------------------------------------------------------------
# Low-quality output filter (internal)
# ---------------------------------------------------------------------------


def _is_low_quality_output(instr: Instruction) -> bool:
    """Return True if the output should be rejected as low quality.

    Criteria:
    - Output is empty or fewer than 3 characters.
    - Output is nearly identical to the instruction (verbatim copy).
    """
    out = instr.output.strip()
    if len(out) < 3:
        return True
    # Reject if output looks like a verbatim copy of the instruction
    if rouge_l_similarity(out, instr.instruction) > 0.9:
        return True
    return False


# ---------------------------------------------------------------------------
# Seed instructions
# ---------------------------------------------------------------------------


def make_seed_instructions() -> list[Instruction]:
    """Return 20 diverse seed instructions for testing and bootstrapping."""
    seeds = [
        Instruction(
            instruction="Write a short poem about the changing seasons.",
            source="seed",
        ),
        Instruction(
            instruction="Summarize the following paragraph in one sentence.",
            source="seed",
        ),
        Instruction(
            instruction="Translate the following sentence into French.",
            source="seed",
        ),
        Instruction(
            instruction="Explain the concept of machine learning to a 10-year-old.",
            source="seed",
        ),
        Instruction(
            instruction="Is this movie review positive or negative?",
            is_classification=True,
            source="seed",
        ),
        Instruction(
            instruction="Write a Python function that computes the factorial of n.",
            source="seed",
        ),
        Instruction(
            instruction="List five healthy breakfast ideas.",
            source="seed",
        ),
        Instruction(
            instruction="Classify the following email as spam or not spam.",
            is_classification=True,
            source="seed",
        ),
        Instruction(
            instruction="Generate a creative title for an article about space exploration.",
            source="seed",
        ),
        Instruction(
            instruction="Rewrite the following sentence to make it more formal.",
            source="seed",
        ),
        Instruction(
            instruction="Solve the following algebra problem step by step.",
            source="seed",
        ),
        Instruction(
            instruction="Describe the main differences between TCP and UDP.",
            source="seed",
        ),
        Instruction(
            instruction="Write a SQL query to find the top 10 most expensive products.",
            source="seed",
        ),
        Instruction(
            instruction="Determine whether the following statement is true or false.",
            is_classification=True,
            source="seed",
        ),
        Instruction(
            instruction="Give three reasons why regular exercise is beneficial.",
            source="seed",
        ),
        Instruction(
            instruction="Paraphrase the following sentence without changing its meaning.",
            source="seed",
        ),
        Instruction(
            instruction="What are the key ingredients for a classic carbonara pasta?",
            source="seed",
        ),
        Instruction(
            instruction="Identify whether the following text expresses positive, neutral, or negative sentiment.",  # noqa: E501
            is_classification=True,
            source="seed",
        ),
        Instruction(
            instruction="Write a cover letter for a software engineering internship.",
            source="seed",
        ),
        Instruction(
            instruction="Explain the water cycle in simple terms.",
            source="seed",
        ),
    ]
    return seeds
