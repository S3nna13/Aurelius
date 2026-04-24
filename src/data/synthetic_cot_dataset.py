from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CoTExample:
    question: str
    chain_of_thought: str
    answer: str
    domain: str = "general"


@dataclass
class CoTDatasetConfig:
    n_examples: int = 100
    domains: list = field(default_factory=lambda: ["math", "logic", "coding", "general"])
    cot_style: str = "numbered"
    min_steps: int = 2
    max_steps: int = 5


class SyntheticCoTDataset:
    def __init__(self, config: Optional[CoTDatasetConfig] = None) -> None:
        self.config = config if config is not None else CoTDatasetConfig()

    def generate_math_example(self, seed: int = 0) -> CoTExample:
        rng = random.Random(seed)
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        c = rng.randint(1, 10)
        question = f"A store has {a} apples. They receive {b} more and then sell {c}. How many apples remain?"
        total = a + b
        result = total - c
        cot = (
            f"1. Start with {a} apples.\n"
            f"2. Add {b} more: {a} + {b} = {total}.\n"
            f"3. Subtract {c} sold: {total} - {c} = {result}."
        )
        return CoTExample(
            question=question,
            chain_of_thought=cot,
            answer=str(result),
            domain="math",
        )

    def generate_logic_example(self, seed: int = 0) -> CoTExample:
        rng = random.Random(seed)
        subjects = ["All cats", "All dogs", "All birds", "All fish"]
        predicates = ["are animals", "can move", "need food", "have cells"]
        subj = subjects[rng.randint(0, len(subjects) - 1)]
        pred = predicates[rng.randint(0, len(predicates) - 1)]
        specific = subj.split()[-1][:-1].capitalize()
        name = ["Luna", "Max", "Bella", "Charlie"][rng.randint(0, 3)]
        question = f"{subj} {pred}. {name} is a {specific.lower()}. Does {name} {pred.split(' ', 1)[1] if ' ' in pred else pred}?"
        cot = (
            f"1. Premise 1: {subj} {pred}.\n"
            f"2. Premise 2: {name} is a {specific.lower()}.\n"
            f"3. By universal instantiation, {name} {pred}."
        )
        answer = "Yes"
        return CoTExample(
            question=question,
            chain_of_thought=cot,
            answer=answer,
            domain="logic",
        )

    def generate_coding_example(self, seed: int = 0) -> CoTExample:
        rng = random.Random(seed)
        n = rng.randint(1, 8)
        question = f"What is the output of: print(sum(range({n})))?"
        total = sum(range(n))
        cot = (
            f"1. range({n}) produces numbers 0 through {n - 1}.\n"
            f"2. sum of 0..{n - 1} = {total}.\n"
            f"3. print outputs {total}."
        )
        return CoTExample(
            question=question,
            chain_of_thought=cot,
            answer=str(total),
            domain="coding",
        )

    def generate_general_example(self, seed: int = 0) -> CoTExample:
        rng = random.Random(seed)
        topics = [
            ("How many days are in 3 weeks?", 7, 3, "days/week", "weeks"),
            ("How many hours are in 2 days?", 24, 2, "hours/day", "days"),
            ("How many minutes are in 4 hours?", 60, 4, "minutes/hour", "hours"),
        ]
        idx = rng.randint(0, len(topics) - 1)
        question, unit, count, rate, unit_name = topics[idx]
        result = unit * count
        cot = (
            f"1. There are {unit} {rate}.\n"
            f"2. We have {count} {unit_name}.\n"
            f"3. Total = {unit} × {count} = {result}."
        )
        return CoTExample(
            question=question,
            chain_of_thought=cot,
            answer=str(result),
            domain="general",
        )

    def generate_example(self, domain: str, seed: int = 0) -> CoTExample:
        if domain == "math":
            return self.generate_math_example(seed)
        if domain == "logic":
            return self.generate_logic_example(seed)
        if domain == "coding":
            return self.generate_coding_example(seed)
        return self.generate_general_example(seed)

    def generate(self, n: Optional[int] = None) -> List[CoTExample]:
        count = n if n is not None else self.config.n_examples
        domains = self.config.domains
        examples: List[CoTExample] = []
        for i in range(count):
            domain = domains[i % len(domains)]
            seed = i // len(domains)
            examples.append(self.generate_example(domain, seed=seed))
        return examples

    def to_chatml(self, example: CoTExample) -> str:
        return (
            f"<|user|>\n{example.question}<|end|>\n"
            f"<|assistant|>\n{example.chain_of_thought}\n\nAnswer: {example.answer}<|end|>"
        )

    def export_jsonl(self, examples: List[CoTExample], path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "question": ex.question,
                    "chain_of_thought": ex.chain_of_thought,
                    "answer": ex.answer,
                    "domain": ex.domain,
                }
                f.write(json.dumps(record) + "\n")


COT_DATASET_REGISTRY: dict = {"default": SyntheticCoTDataset}
