"""
instruction_synthesis.py — Instruction synthesis for Aurelius LLM training data.

Generates diverse training examples from templates and seed tasks using
pure Python stdlib only (no PyTorch, HuggingFace, or NLTK required).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SynthConfig:
    """Configuration for instruction synthesis."""

    n_instructions_per_seed: int = 10
    similarity_threshold: float = 0.7
    max_length: int = 512
    min_length: int = 20
    templates: List[str] = field(
        default_factory=lambda: [
            "Explain {topic}",
            "Write a {adjective} {document_type} about {topic}",
            "List {n} ways to {action}",
            "Compare {a} and {b}",
        ]
    )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def fill_template(template: str, slots: Dict[str, str]) -> str:
    """Replace {slot} placeholders with values from *slots*.

    Any placeholder not present in *slots* is left unchanged.
    """
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        return slots.get(key, match.group(0))

    return re.sub(r"\{(\w+)\}", replacer, template)


def compute_rouge1_similarity(a: str, b: str) -> float:
    """Compute ROUGE-1 F1 (unigram overlap) between two strings.

    Tokenises on whitespace; returns 1.0 for two empty strings.
    """
    tokens_a = a.lower().split()
    tokens_b = b.lower().split()

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)
    overlap = len(set_a & set_b)

    precision = overlap / len(set_b)
    recall = overlap / len(set_a)

    if precision + recall == 0.0:
        return 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def deduplicate_instructions(
    instructions: List[str], threshold: float = 0.7
) -> List[str]:
    """Remove instructions whose ROUGE-1 similarity to any earlier kept
    instruction is >= *threshold*.  Preserves order of first occurrence.
    """
    kept: List[str] = []
    for candidate in instructions:
        duplicate = False
        for existing in kept:
            if compute_rouge1_similarity(candidate, existing) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def classify_instruction_type(instruction: str) -> str:
    """Heuristic classification of an instruction string.

    Priority order:
    1. open_qa   — starts with who/what/when/where/why/how
    2. classification — contains classify/categorize/label
    3. generation    — contains write/create/generate
    4. analysis      — contains analyze/compare/explain
    5. other
    """
    lower = instruction.lower().strip()

    # open_qa: starts with a question word
    if re.match(r"^(who|what|when|where|why|how)\b", lower):
        return "open_qa"

    # classification
    if re.search(r"\b(classify|categorize|label)\b", lower):
        return "classification"

    # generation
    if re.search(r"\b(write|create|generate)\b", lower):
        return "generation"

    # analysis
    if re.search(r"\b(analyze|compare|explain)\b", lower):
        return "analysis"

    return "other"


def estimate_difficulty(instruction: str, max_length: int = 512) -> float:
    """Estimate difficulty in [0, 1].

    Heuristic: longer text = harder; more commas = harder.
    Score = clip(length/max_length + n_commas/10, 0, 1).
    """
    length_score = len(instruction) / max_length
    comma_score = instruction.count(",") / 10.0
    raw = length_score + comma_score
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Synthesiser class
# ---------------------------------------------------------------------------

class InstructionSynthesizer:
    """Generates diverse instructions from templates and seed tasks."""

    def __init__(self, config: SynthConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_template(
        self, template: str, slots_list: List[Dict[str, str]]
    ) -> List[str]:
        """Fill *template* with each slots dict and return valid instructions.

        Validity: length between config.min_length and config.max_length.
        """
        results: List[str] = []
        for slots in slots_list:
            filled = fill_template(template, slots)
            if self.config.min_length <= len(filled) <= self.config.max_length:
                results.append(filled)
        return results

    def from_seed(
        self, seed_instruction: str, variations: List[Dict[str, str]]
    ) -> List[str]:
        """Generate variations of *seed_instruction*.

        For each variation dict, replace every key found verbatim in the seed
        with its corresponding value.  Returns up to
        config.n_instructions_per_seed results (seed itself is excluded).
        """
        generated: List[str] = []
        for var in variations:
            modified = seed_instruction
            for word, replacement in var.items():
                modified = modified.replace(word, replacement)
            if modified != seed_instruction:
                generated.append(modified)
            if len(generated) >= self.config.n_instructions_per_seed:
                break
        return generated

    def build_dataset(
        self,
        seeds: List[str],
        slots_list: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Build a dataset of instruction dicts.

        For each seed:
        - Treat *slots_list* as variations (key→value replacements).
        - Generate variants via from_seed.
        - Include the seed itself.
        - Deduplicate, classify, estimate difficulty.

        Returns list of {"instruction", "type", "difficulty"} dicts.
        """
        all_instructions: List[str] = []
        for seed in seeds:
            candidates = [seed] + self.from_seed(seed, slots_list)
            candidates = deduplicate_instructions(
                candidates, self.config.similarity_threshold
            )
            all_instructions.extend(candidates)

        # Global deduplication across seeds
        all_instructions = deduplicate_instructions(
            all_instructions, self.config.similarity_threshold
        )

        dataset: List[Dict[str, str]] = []
        for instr in all_instructions:
            dataset.append(
                {
                    "instruction": instr,
                    "type": classify_instruction_type(instr),
                    "difficulty": estimate_difficulty(
                        instr, self.config.max_length
                    ),
                }
            )
        return dataset

    def get_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics for a dataset.

        Returns dict with keys:
        - n_total
        - type_distribution (counts per type)
        - mean_difficulty
        - mean_length
        """
        n_total = len(dataset)
        type_dist: Dict[str, int] = {}
        total_difficulty = 0.0
        total_length = 0

        for item in dataset:
            t = item["type"]
            type_dist[t] = type_dist.get(t, 0) + 1
            total_difficulty += item["difficulty"]
            total_length += len(item["instruction"])

        mean_difficulty = total_difficulty / n_total if n_total else 0.0
        mean_length = total_length / n_total if n_total else 0.0

        return {
            "n_total": n_total,
            "type_distribution": type_dist,
            "mean_difficulty": mean_difficulty,
            "mean_length": mean_length,
        }
