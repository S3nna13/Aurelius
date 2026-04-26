"""Synthetic instruction data generation: Self-Instruct and Evol-Instruct style templates."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class InstructConfig:
    """Configuration for synthetic instruction generation."""

    n_instructions: int = 100
    max_instruction_len: int = 200  # max chars
    complexity_levels: int = 3  # 1=simple, 2=medium, 3=complex
    domains: list[str] = field(
        default_factory=lambda: ["math", "coding", "reasoning", "writing", "factual"]
    )
    seed: int = 42
    include_cot: bool = True  # include chain-of-thought in answer


@dataclass
class InstructionSample:
    """A single instruction-following training sample."""

    instruction: str
    input_context: str  # optional context/input
    output: str  # expected output
    domain: str
    complexity: int
    has_cot: bool


# ---------------------------------------------------------------------------
# Domain generators
# ---------------------------------------------------------------------------


def generate_math_instruction(
    rng: random.Random, complexity: int, include_cot: bool = True
) -> InstructionSample:
    """Generate a math instruction sample at the given complexity level."""
    if complexity == 1:
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(["+", "-", "*"])
        instruction = f"What is {a} {op} {b}?"
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        if include_cot:
            output = f"Step-by-step: {a} {op} {b} = {result}\nThe answer is {result}."
        else:
            output = str(result)

    elif complexity == 2:
        x_val = rng.randint(1, 20)
        y_val = rng.randint(1, 20)
        a_coeff = rng.randint(1, 5)
        b_coeff = rng.randint(1, 5)
        instruction = f"If x = {x_val} and y = {y_val}, what is {a_coeff}x + {b_coeff}y?"
        result = a_coeff * x_val + b_coeff * y_val
        if include_cot:
            output = (
                f"Step-by-step: {a_coeff}x + {b_coeff}y = "
                f"{a_coeff}*{x_val} + {b_coeff}*{y_val} = "
                f"{a_coeff * x_val} + {b_coeff * y_val} = {result}\n"
                f"The answer is {result}."
            )
        else:
            output = str(result)

    else:  # complexity >= 3
        total = rng.randint(20, 200)
        pct = rng.choice([10, 20, 25, 30, 40, 50])
        instruction = f"A store has {total} items. {pct}% are sold. How many remain?"
        sold = total * pct // 100
        remain = total - sold
        if include_cot:
            output = (
                f"Step-by-step: {pct}% of {total} = {sold} items sold. "
                f"{total} - {sold} = {remain} items remain.\n"
                f"The answer is {remain}."
            )
        else:
            output = str(remain)

    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        domain="math",
        complexity=complexity,
        has_cot=include_cot,
    )


def generate_coding_instruction(
    rng: random.Random, complexity: int, include_cot: bool = True
) -> InstructionSample:
    """Generate a coding instruction sample at the given complexity level."""
    if complexity == 1:
        tasks = [
            (
                "Write a Python function that returns the sum of a list of numbers.",
                "def sum_list(nums):\n    return sum(nums)",
            ),
            (
                "Write a Python function that returns the length of a string.",
                "def string_length(s):\n    return len(s)",
            ),
            (
                "Write a Python function that reverses a string.",
                "def reverse_string(s):\n    return s[::-1]",
            ),
            (
                "Write a Python function that returns the maximum value in a list.",
                "def max_value(nums):\n    return max(nums)",
            ),
        ]
    elif complexity == 2:
        tasks = [
            (
                "Write a Python function that finds the nth Fibonacci number.",
                (
                    "def fibonacci(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    a, b = 0, 1\n"
                    "    for _ in range(2, n + 1):\n"
                    "        a, b = b, a + b\n"
                    "    return b"
                ),
            ),
            (
                "Write a Python function that checks if a string is a palindrome.",
                "def is_palindrome(s):\n    return s == s[::-1]",
            ),
            (
                "Write a Python function that removes duplicates from a list while preserving order.",  # noqa: E501
                (
                    "def remove_duplicates(lst):\n"
                    "    seen = set()\n"
                    "    result = []\n"
                    "    for item in lst:\n"
                    "        if item not in seen:\n"
                    "            seen.add(item)\n"
                    "            result.append(item)\n"
                    "    return result"
                ),
            ),
        ]
    else:  # complexity >= 3
        tasks = [
            (
                "Implement a binary search function on a sorted list that returns the index of the target, or -1 if not found.",  # noqa: E501
                (
                    "def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid - 1\n"
                    "    return -1"
                ),
            ),
            (
                "Implement a merge sort function that sorts a list in ascending order.",
                (
                    "def merge_sort(arr):\n"
                    "    if len(arr) <= 1:\n"
                    "        return arr\n"
                    "    mid = len(arr) // 2\n"
                    "    left = merge_sort(arr[:mid])\n"
                    "    right = merge_sort(arr[mid:])\n"
                    "    return merge(left, right)\n\n"
                    "def merge(left, right):\n"
                    "    result = []\n"
                    "    i = j = 0\n"
                    "    while i < len(left) and j < len(right):\n"
                    "        if left[i] <= right[j]:\n"
                    "            result.append(left[i])\n"
                    "            i += 1\n"
                    "        else:\n"
                    "            result.append(right[j])\n"
                    "            j += 1\n"
                    "    result.extend(left[i:])\n"
                    "    result.extend(right[j:])\n"
                    "    return result"
                ),
            ),
        ]

    instruction, solution = rng.choice(tasks)
    if include_cot:
        output = f"Here is the solution:\n\n```python\n{solution}\n```"
    else:
        output = solution

    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        domain="coding",
        complexity=complexity,
        has_cot=include_cot,
    )


def generate_reasoning_instruction(
    rng: random.Random, complexity: int, include_cot: bool = True
) -> InstructionSample:
    """Generate a reasoning instruction sample (logic, sequences, analogies)."""
    if complexity == 1:
        # Sequence completion
        start = rng.randint(1, 10)
        step = rng.randint(2, 5)
        seq = [start + step * i for i in range(4)]
        answer = start + step * 4
        instruction = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
        if include_cot:
            output = (
                f"Step-by-step: Each number increases by {step}. "
                f"{seq[-1]} + {step} = {answer}\nThe answer is {answer}."
            )
        else:
            output = str(answer)

    elif complexity == 2:
        # Simple logic
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        n1, n2 = rng.sample(names, 2)
        age1 = rng.randint(20, 50)
        diff = rng.randint(1, 15)
        instruction = (
            f"{n1} is {age1} years old. {n2} is {diff} years older than {n1}. How old is {n2}?"
        )
        answer = age1 + diff
        if include_cot:
            output = (
                f"Step-by-step: {n2}'s age = {n1}'s age + {diff} = {age1} + {diff} = {answer}\n"
                f"The answer is {answer}."
            )
        else:
            output = str(answer)

    else:  # complexity >= 3
        # Multi-step logic puzzle
        names = ["Alice", "Bob", "Carol"]
        items = rng.sample(["apple", "banana", "cherry", "date", "elderberry"], 3)
        assignment = list(zip(names, items))
        rng.shuffle(assignment)
        clues = []
        for name, item in assignment:
            clues.append(f"{name} picked the {item}.")
        ask_name = assignment[0][0]
        ask_item = assignment[0][1]
        instruction = (
            "Given the following clues:\n" + "\n".join(clues) + f"\nWhat did {ask_name} pick?"
        )
        if include_cot:
            output = (
                f"Step-by-step: From the clues, {ask_name} picked the {ask_item}.\n"
                f"The answer is {ask_item}."
            )
        else:
            output = ask_item

    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        domain="reasoning",
        complexity=complexity,
        has_cot=include_cot,
    )


def generate_writing_instruction(
    rng: random.Random, complexity: int, include_cot: bool = True
) -> InstructionSample:
    """Generate a writing instruction sample."""
    if complexity == 1:
        topics = ["the water cycle", "photosynthesis", "gravity", "the solar system"]
        topic = rng.choice(topics)
        instruction = f"Explain {topic} in one or two simple sentences."
        output = f"{topic.capitalize()} is a natural process. It is important to understand its basic principles."  # noqa: E501

    elif complexity == 2:
        topics = [
            ("machine learning", "artificial intelligence"),
            ("democracy", "government"),
            ("evolution", "biology"),
            ("climate change", "environment"),
        ]
        topic, field_name = rng.choice(topics)
        instruction = (
            f"Write a short paragraph explaining {topic} and its significance in {field_name}."
        )
        output = (
            f"{topic.capitalize()} is a key concept in {field_name}. "
            f"Understanding {topic} helps us appreciate the broader field of {field_name} "
            f"and its impact on society."
        )

    else:  # complexity >= 3
        tasks = [
            (
                "Summarize the pros and cons of remote work in a balanced short essay.",
                (
                    "Remote work offers flexibility and eliminates commuting, "
                    "but can lead to isolation and blurred work-life boundaries. "
                    "Organizations must weigh productivity gains against collaboration challenges."
                ),
            ),
            (
                "Rewrite the following sentence to be more concise: "
                "'The reason why the project was not completed on time is because there were not enough resources available.'",  # noqa: E501
                "The project missed its deadline due to insufficient resources.",
            ),
            (
                "Write a persuasive paragraph arguing for renewable energy adoption.",
                (
                    "Renewable energy sources like solar and wind offer sustainable alternatives "
                    "to fossil fuels. Transitioning to renewables reduces carbon emissions, "
                    "creates jobs, and ensures energy security for future generations."
                ),
            ),
        ]
        instruction, output = rng.choice(tasks)

    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        domain="writing",
        complexity=complexity,
        has_cot=include_cot,
    )


def generate_factual_instruction(
    rng: random.Random, complexity: int, include_cot: bool = True
) -> InstructionSample:
    """Generate a factual instruction sample (definitions, comparisons, classifications)."""
    if complexity == 1:
        terms = [
            (
                "algorithm",
                "A step-by-step procedure for solving a problem or performing a computation.",
            ),
            (
                "photosynthesis",
                "The process by which plants convert sunlight into chemical energy.",
            ),
            ("democracy", "A system of government where power is vested in the people."),
            ("gravity", "A force that attracts objects with mass toward each other."),
        ]
        term, definition = rng.choice(terms)
        instruction = f"Define the term '{term}'."
        output = definition

    elif complexity == 2:
        comparisons = [
            (
                "CPU",
                "GPU",
                "A CPU (Central Processing Unit) handles general-purpose tasks with a few powerful cores. "  # noqa: E501
                "A GPU (Graphics Processing Unit) has many smaller cores optimized for parallel computation.",  # noqa: E501
            ),
            (
                "TCP",
                "UDP",
                "TCP provides reliable, ordered delivery with error checking. "
                "UDP is faster but does not guarantee delivery or ordering.",
            ),
            (
                "compiler",
                "interpreter",
                "A compiler translates the entire source code to machine code before execution. "
                "An interpreter executes code line by line at runtime.",
            ),
        ]
        t1, t2, answer = rng.choice(comparisons)
        instruction = f"Compare and contrast {t1} and {t2}."
        output = answer

    else:  # complexity >= 3
        tasks = [
            (
                "Classify the following into programming paradigms: Python, Haskell, C, Java, Prolog.",  # noqa: E501
                (
                    "Multi-paradigm: Python, Java. "
                    "Functional: Haskell. "
                    "Procedural/Imperative: C. "
                    "Logic: Prolog."
                ),
            ),
            (
                "Classify these data structures by access pattern: array, linked list, hash table, binary search tree.",  # noqa: E501
                (
                    "Random access: array, hash table. "
                    "Sequential access: linked list. "
                    "Ordered access: binary search tree."
                ),
            ),
            (
                "Classify these sorting algorithms by time complexity (average case): bubble sort, merge sort, quicksort, insertion sort.",  # noqa: E501
                ("O(n log n): merge sort, quicksort. O(n^2): bubble sort, insertion sort."),
            ),
        ]
        instruction, output = rng.choice(tasks)

    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        domain="factual",
        complexity=complexity,
        has_cot=include_cot,
    )


# ---------------------------------------------------------------------------
# Evol-Instruct
# ---------------------------------------------------------------------------


def evolve_instruction(rng: random.Random, base: InstructionSample) -> InstructionSample:
    """Evol-Instruct: make an instruction harder by adding constraints or combining steps.

    Returns a new InstructionSample with increased complexity (capped at 3).
    """
    evolution_strategies = [
        "Add a constraint: ",
        "Extend the task: ",
        "Require explanation: ",
    ]
    rng.choice(evolution_strategies)

    constraints = [
        "Also explain your reasoning.",
        "Do this without using any built-in functions.",
        "Provide the answer in exactly two sentences.",
        "Include at least one example.",
        "Solve this in the most efficient way possible.",
    ]
    constraint = rng.choice(constraints)

    new_complexity = min(base.complexity + 1, 3)
    new_instruction = f"{base.instruction} {constraint}"
    new_output = f"{base.output}\n\n[Evolved: {constraint}]"

    return InstructionSample(
        instruction=new_instruction,
        input_context=base.input_context,
        output=new_output,
        domain=base.domain,
        complexity=new_complexity,
        has_cot=base.has_cot,
    )


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

_DOMAIN_GENERATORS = {
    "math": generate_math_instruction,
    "coding": generate_coding_instruction,
    "reasoning": generate_reasoning_instruction,
    "writing": generate_writing_instruction,
    "factual": generate_factual_instruction,
}


class InstructionGenerator:
    """Generates synthetic instruction-following training data."""

    def __init__(self, config: InstructConfig | None = None) -> None:
        self.config = config or InstructConfig()
        self._rng = random.Random(self.config.seed)

    def generate(self) -> list[InstructionSample]:
        """Generate n_instructions samples distributed across domains and complexity levels."""
        cfg = self.config
        samples: list[InstructionSample] = []
        domains = [d for d in cfg.domains if d in _DOMAIN_GENERATORS]

        for i in range(cfg.n_instructions):
            domain = domains[i % len(domains)]
            complexity = (i % cfg.complexity_levels) + 1
            gen_fn = _DOMAIN_GENERATORS[domain]
            sample = gen_fn(self._rng, complexity, cfg.include_cot)

            # Truncate instruction to max_instruction_len
            if len(sample.instruction) > cfg.max_instruction_len:
                sample = InstructionSample(
                    instruction=sample.instruction[: cfg.max_instruction_len],
                    input_context=sample.input_context,
                    output=sample.output,
                    domain=sample.domain,
                    complexity=sample.complexity,
                    has_cot=sample.has_cot,
                )

            # Optionally evolve some instructions (every 5th)
            if i % 5 == 4:
                sample = evolve_instruction(self._rng, sample)

            samples.append(sample)

        return samples

    def generate_batch(self, n: int, domain: str | None = None) -> list[InstructionSample]:
        """Generate n samples, optionally filtered by domain."""
        cfg = self.config
        samples: list[InstructionSample] = []

        if domain is not None and domain in _DOMAIN_GENERATORS:
            gen_fn = _DOMAIN_GENERATORS[domain]
            for i in range(n):
                complexity = (i % cfg.complexity_levels) + 1
                sample = gen_fn(self._rng, complexity, cfg.include_cot)
                samples.append(sample)
        else:
            domains = [d for d in cfg.domains if d in _DOMAIN_GENERATORS]
            for i in range(n):
                d = domains[i % len(domains)]
                complexity = (i % cfg.complexity_levels) + 1
                gen_fn = _DOMAIN_GENERATORS[d]
                sample = gen_fn(self._rng, complexity, cfg.include_cot)
                samples.append(sample)

        return samples

    def to_sft_format(self, samples: list[InstructionSample]) -> list[dict]:
        """Convert samples to SFT format: [{"prompt": str, "completion": str}, ...]."""
        results = []
        for s in samples:
            prompt = s.instruction
            if s.input_context:
                prompt = f"{s.instruction}\n\nContext: {s.input_context}"
            results.append({"prompt": prompt, "completion": s.output})
        return results
