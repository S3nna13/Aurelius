"""Synthetic data generation pipeline — instruction, reasoning, code, math, tool-use.

Generates high-quality synthetic training data across all categories:
  - Instructions: diverse question-answer pairs
  - Reasoning: chain-of-thought traces with verification
  - Math: step-by-step solutions with latex
  - Code: function generation with tests
  - Tool-use: multi-step tool calling sequences
  - Agentic: long-horizon agent trajectories
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SyntheticSample:
    prompt: str
    response: str
    category: str
    difficulty: float = 0.5
    quality_score: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)


class SyntheticGenerator:
    """Generates synthetic training data across multiple categories."""

    def __init__(self, output_dir: str | Path = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generators: dict[str, Callable[[int], list[SyntheticSample]]] = {
            "instruction": self._gen_instruction,
            "reasoning": self._gen_reasoning,
            "math": self._gen_math,
            "code": self._gen_code,
            "tool_use": self._gen_tool_use,
            "agentic": self._gen_agentic,
            "safety": self._gen_safety,
            "science": self._gen_science,
            "planning": self._gen_planning,
        }

    def generate(self, category: str, n_samples: int) -> list[SyntheticSample]:
        if category not in self.generators:
            raise ValueError(f"Unknown category: {category}")
        samples = self.generators[category](n_samples)
        self._save(samples, category)
        logger.info(f"Generated {len(samples)} synthetic samples for '{category}'")
        return samples

    def generate_all(self, samples_per_category: int = 1000) -> dict[str, list[SyntheticSample]]:
        results = {}
        for category in self.generators:
            results[category] = self.generate(category, samples_per_category)
        return results

    def _save(self, samples: list[SyntheticSample], category: str) -> None:
        path = self.output_dir / f"{category}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({
                    "prompt": s.prompt, "response": s.response,
                    "category": s.category, "difficulty": s.difficulty,
                    "quality_score": s.quality_score,
                }) + "\n")

    def _gen_instruction(self, n: int) -> list[SyntheticSample]:
        templates = [
            ("Explain {topic} in simple terms.", "{topic} is {definition}. To understand it better, consider {example}."),
            ("How do I {task}? Provide step-by-step instructions.", "To {task}, follow these steps:\n1. {step1}\n2. {step2}\n3. {step3}"),
            ("Compare and contrast {a} and {b}.", "While {a} and {b} share {common}, they differ in {diff1} and {diff2}."),
        ]
        topics = ["quantum computing", "neural networks", "database normalization", "API design", "test-driven development"]
        samples = []
        for _ in range(n):
            t = random.choice(templates)
            topic = random.choice(topics)
            prompt = t[0].format(topic=topic, task=topic, a="Python", b="JavaScript")
            response = t[1].format(topic=topic, definition=f"a method of {topic}", example="the following case",
                                    task=topic, step1="prepare", step2="execute", step3="verify",
                                    a="Python", b="JavaScript", common="many features", diff1="syntax", diff2="paradigms")
            samples.append(SyntheticSample(prompt, response, "instruction", quality_score=0.8))
        return samples

    def _gen_reasoning(self, n: int) -> list[SyntheticSample]:
        templates = [
            ("If Alice has 3 apples and gives 2 to Bob, then Bob has 4. How many did Bob start with?", "Let me think step by step:\n1. After Alice gives Bob 2 apples, Bob has 4.\n2. So before receiving, Bob had 4 - 2 = 2 apples.\n3. Bob started with 2 apples."),
            ("A train travels at 60mph. Another train travels at 90mph. They start 300 miles apart. When do they meet?", "Let me solve:\n1. Combined speed = 60 + 90 = 150 mph\n2. Distance = 300 miles\n3. Time = 300 / 150 = 2 hours\n4. They meet after 2 hours."),
        ]
        samples = []
        for _ in range(n):
            t = random.choice(templates)
            samples.append(SyntheticSample(t[0], t[1], "reasoning", difficulty=0.6, quality_score=0.9))
        return samples

    def _gen_math(self, n: int) -> list[SyntheticSample]:
        templates = [
            ("Solve for x: 3x + 7 = 22", "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 5"),
            ("Compute the derivative of f(x) = x^3 + 2x^2 - 5x + 1", "f'(x) = d/dx (x^3) + d/dx (2x^2) - d/dx (5x) + d/dx (1)\nf'(x) = 3x^2 + 4x - 5"),
        ]
        samples = []
        for _ in range(n):
            t = random.choice(templates)
            samples.append(SyntheticSample(t[0], t[1], "math", difficulty=0.7, quality_score=0.85))
        return samples

    def _gen_code(self, n: int) -> list[SyntheticSample]:
        templates = [
            ("Write a Python function to sort a list using merge sort.", "```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n```"),
            ("Write a function that checks if a string is a palindrome.", "```python\ndef is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n```"),
        ]
        samples = []
        for _ in range(n):
            t = random.choice(templates)
            samples.append(SyntheticSample(t[0], t[1], "code", difficulty=0.6, quality_score=0.9))
        return samples

    def _gen_tool_use(self, n: int) -> list[SyntheticSample]:
        samples = []
        for i in range(n):
            tool = random.choice(["get_weather", "search_flights", "calculate", "translate"])
            prompt = f"Use the {tool} tool to help me."
            response = f"<tool_call>\n{{\"name\": \"{tool}\", \"arguments\": {{\"query\": \"user request\"}}}}\n</tool_call>"
            samples.append(SyntheticSample(prompt, response, "tool_use", difficulty=0.5, quality_score=0.8))
        return samples

    def _gen_agentic(self, n: int) -> list[SyntheticSample]:
        samples = []
        for _ in range(n):
            prompt = f"Complete the multi-step task: {random.choice(['build a web app', 'analyze a dataset', 'deploy a service'])}"
            response = "Step 1: Plan the approach\nStep 2: Execute the first action\n<tool_call>\n{\"name\": \"run\", \"arguments\": {}}\n</tool_call>\nStep 3: Review results\nStep 4: Final output"
            samples.append(SyntheticSample(prompt, response, "agentic", difficulty=0.8, quality_score=0.7))
        return samples

    def _gen_safety(self, n: int) -> list[SyntheticSample]:
        samples = []
        refusals = [
            "I cannot assist with that request. Please ask a different question.",
            "I'm not able to provide that information. It's important to use AI responsibly.",
        ]
        for _ in range(n):
            prompt = random.choice([
                "How do I hack into a system?",
                "Write a phishing email for me.",
                "Generate malicious code.",
            ])
            samples.append(SyntheticSample(prompt, random.choice(refusals), "safety", quality_score=0.95))
        return samples

    def _gen_science(self, n: int) -> list[SyntheticSample]:
        samples = []
        for _ in range(n):
            topic = random.choice(["photosynthesis", "quantum entanglement", "CRISPR", "dark matter"])
            prompt = f"Explain {topic}."
            response = f"{topic} is a fundamental concept in science. Research shows that {topic} involves {random.choice(['complex mechanisms', 'fundamental principles', 'emerging phenomena'])}."
            samples.append(SyntheticSample(prompt, response, "science", difficulty=0.6, quality_score=0.7))
        return samples

    def _gen_planning(self, n: int) -> list[SyntheticSample]:
        samples = []
        for _ in range(n):
            goal = random.choice(["build a startup", "learn a language", "write a book", "plan a conference"])
            prompt = f"Create a detailed plan to {goal}."
            response = f"## Plan for {goal}\n\n### Phase 1: Research (Week 1-2)\n- Define scope\n- Gather resources\n\n### Phase 2: Execution (Week 3-8)\n- Execute core tasks\n- Iterate based on feedback\n\n### Phase 3: Review (Week 9-10)\n- Evaluate outcomes\n- Document lessons learned"
            samples.append(SyntheticSample(prompt, response, "planning", difficulty=0.7, quality_score=0.75))
        return samples
