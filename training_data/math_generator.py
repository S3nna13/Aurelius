# Aurelius Training Data — Math Generator
# Licensed MIT — Christien Antonio, 2026
"""Synthetic math and reasoning training data generator in JSONL format.

Output schema:
    {"instruction": str, "response": str, "category": str, "difficulty": str}

Categories:
    arithmetic, algebra, geometry, probability, word_problems, formal_proofs

Difficulties:
    easy   — 1-2 steps, straightforward
    medium — 3-5 steps, requires multiple concepts
    hard   — 5+ steps, requires insight or combination of techniques
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


class MathDataGenerator:
    """Generates synthetic math/reasoning training data with step-by-step solutions."""

    CATEGORIES = [
        "arithmetic",
        "algebra",
        "geometry",
        "probability",
        "word_problems",
        "formal_proofs",
    ]
    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        seed = config.get("seed", 42)
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_problem(self, category: str, difficulty: str) -> dict[str, str]:
        if category not in self.CATEGORIES:
            msg = f"Unknown category: {category}. Choose from {self.CATEGORIES}"
            raise ValueError(msg)
        if difficulty not in self.DIFFICULTIES:
            msg = f"Unknown difficulty: {difficulty}. Choose from {self.DIFFICULTIES}"
            raise ValueError(msg)

        method_name = f"_generate_{category}"
        method = getattr(self, method_name)
        instruction, response = method(difficulty)

        return {
            "instruction": instruction,
            "response": response,
            "category": category,
            "difficulty": difficulty,
        }

    def run(self, num_samples: int, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        math_dir = output_dir / "math"
        math_dir.mkdir(parents=True, exist_ok=True)

        config_math = self.config.get("math", {})
        type_weights = config_math.get(
            "types",
            {
                "arithmetic": 0.20,
                "algebra": 0.25,
                "geometry": 0.10,
                "probability": 0.15,
                "word_problems": 0.20,
                "formal_proofs": 0.10,
            },
        )
        diff_weights = config_math.get(
            "difficulty",
            {"easy": 0.25, "medium": 0.50, "hard": 0.25},
        )

        categories = list(type_weights.keys())
        cat_weights = [type_weights[c] for c in categories]
        diffs = list(diff_weights.keys())
        diff_w = [diff_weights[d] for d in diffs]

        records: list[dict[str, str]] = []
        for _ in range(num_samples):
            cat = self.rng.choices(categories, weights=cat_weights, k=1)[0]
            dif = self.rng.choices(diffs, weights=diff_w, k=1)[0]
            records.append(self.generate_problem(cat, dif))

        split = int(len(records) * 0.9)
        train_records = records[:split]
        val_records = records[split:]

        with open(math_dir / "train.jsonl", "w") as f:
            for rec in train_records:
                f.write(json.dumps(rec) + "\n")

        with open(math_dir / "val.jsonl", "w") as f:
            for rec in val_records:
                f.write(json.dumps(rec) + "\n")
