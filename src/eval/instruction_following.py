"""Instruction following evaluation: measure format compliance, constraint satisfaction, and instruction adherence."""

import json
import random
import re
from dataclasses import dataclass, field
from statistics import mean


@dataclass
class InstructionConfig:
    eval_format: bool = True
    eval_length: bool = True
    eval_keywords: bool = True
    eval_structure: bool = True
    case_sensitive: bool = False


@dataclass
class Instruction:
    prompt: str
    required_format: str | None = None
    min_words: int | None = None
    max_words: int | None = None
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class ComplianceResult:
    instruction: Instruction
    response: str
    format_score: float
    length_score: float
    keyword_score: float
    structure_score: float
    overall_score: float


def check_format_compliance(response: str, required_format: str | None) -> float:
    if required_format is None:
        return 1.0

    if required_format == "json":
        try:
            json.loads(response)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0

    if required_format == "list":
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        for line in lines:
            if line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.", line):
                return 1.0
        return 0.0

    if required_format == "numbered":
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        for line in lines:
            if re.match(r"^\d+\.", line):
                return 1.0
        return 0.0

    if required_format == "paragraph":
        lines = response.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped and (stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"^\d+\.", stripped)):
                return 0.0
        words = response.split()
        return 1.0 if len(words) >= 20 else 0.0

    return 1.0


def check_length_compliance(response: str, min_words: int | None, max_words: int | None) -> float:
    word_count = len(response.split())
    if min_words is not None and word_count < min_words:
        return 0.0
    if max_words is not None and word_count > max_words:
        return 0.0
    return 1.0


def check_keyword_compliance(
    response: str,
    required: list[str],
    forbidden: list[str],
    case_sensitive: bool = False,
) -> float:
    check_response = response if case_sensitive else response.lower()

    if required:
        present = sum(
            1
            for kw in required
            if (kw if case_sensitive else kw.lower()) in check_response
        )
        required_score = present / len(required)
    else:
        required_score = 1.0

    forbidden_score = 1.0
    for kw in forbidden:
        if (kw if case_sensitive else kw.lower()) in check_response:
            forbidden_score = 0.0
            break

    return min(required_score, forbidden_score)


def check_structure_compliance(response: str, required_format: str | None) -> float:
    if required_format is None:
        return 1.0

    if required_format == "list":
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        items = [
            line for line in lines
            if line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.", line)
        ]
        return 1.0 if len(items) >= 2 else 0.0

    if required_format == "numbered":
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        numbered = []
        for line in lines:
            m = re.match(r"^(\d+)\.", line)
            if m:
                numbered.append(int(m.group(1)))
        if len(numbered) < 2:
            return 0.0
        expected = list(range(1, len(numbered) + 1))
        return 1.0 if numbered == expected else 0.0

    if required_format == "json":
        try:
            obj = json.loads(response)
            if isinstance(obj, dict) and len(obj) >= 1:
                return 1.0
            return 0.0
        except (json.JSONDecodeError, ValueError):
            return 0.0

    if required_format == "paragraph":
        sentences = [s.strip() for s in response.split(". ") if s.strip()]
        return 1.0 if len(sentences) >= 2 else 0.0

    return 1.0


class InstructionFollowingEvaluator:
    def __init__(self, config: InstructionConfig):
        self.config = config

    def evaluate(self, instruction: Instruction, response: str) -> ComplianceResult:
        config = self.config

        format_score = (
            check_format_compliance(response, instruction.required_format)
            if config.eval_format
            else 1.0
        )
        length_score = (
            check_length_compliance(response, instruction.min_words, instruction.max_words)
            if config.eval_length
            else 1.0
        )
        keyword_score = (
            check_keyword_compliance(
                response,
                instruction.required_keywords,
                instruction.forbidden_keywords,
                config.case_sensitive,
            )
            if config.eval_keywords
            else 1.0
        )
        structure_score = (
            check_structure_compliance(response, instruction.required_format)
            if config.eval_structure
            else 1.0
        )

        applicable_scores: list[float] = []
        if config.eval_format:
            applicable_scores.append(format_score)
        if config.eval_length:
            applicable_scores.append(length_score)
        if config.eval_keywords:
            applicable_scores.append(keyword_score)
        if config.eval_structure:
            applicable_scores.append(structure_score)

        overall_score = mean(applicable_scores) if applicable_scores else 1.0

        return ComplianceResult(
            instruction=instruction,
            response=response,
            format_score=format_score,
            length_score=length_score,
            keyword_score=keyword_score,
            structure_score=structure_score,
            overall_score=overall_score,
        )

    def evaluate_batch(
        self, instructions: list[Instruction], responses: list[str]
    ) -> list[ComplianceResult]:
        return [self.evaluate(inst, resp) for inst, resp in zip(instructions, responses)]

    def aggregate_results(self, results: list[ComplianceResult]) -> dict[str, float]:
        if not results:
            return {
                "mean_overall": 0.0,
                "format_rate": 0.0,
                "length_rate": 0.0,
                "keyword_rate": 0.0,
                "structure_rate": 0.0,
            }
        return {
            "mean_overall": mean(r.overall_score for r in results),
            "format_rate": mean(r.format_score for r in results),
            "length_rate": mean(r.length_score for r in results),
            "keyword_rate": mean(r.keyword_score for r in results),
            "structure_rate": mean(r.structure_score for r in results),
        }


def create_instruction_test_suite(n: int = 20, seed: int = 42) -> list[tuple[Instruction, str]]:
    rng = random.Random(seed)
    format_types = ["json", "list", "numbered", "paragraph", None]
    keywords_pool = [
        ["important", "relevant"],
        ["analysis", "data"],
        ["result", "output"],
        ["summary", "overview"],
        ["feature", "model"],
    ]

    suite: list[tuple[Instruction, str]] = []

    gold_responses: dict[str | None, str] = {
        "json": '{"key": "value", "count": 42, "active": true}',
        "list": "- First item in the list\n- Second item in the list\n- Third item in the list",
        "numbered": "1. First step\n2. Second step\n3. Third step",
        "paragraph": (
            "This is the first sentence in a paragraph that contains many words. "
            "It continues with a second sentence that adds more detail. "
            "The paragraph concludes with a final observation about the topic."
        ),
        None: (
            "This is a general response that does not require a specific format. "
            "It contains enough words to satisfy length constraints comfortably."
        ),
    }

    for i in range(n):
        fmt = format_types[i % len(format_types)]
        kw_pair = keywords_pool[i % len(keywords_pool)]
        min_w = rng.choice([None, 10, 15])
        max_w = rng.choice([None, 200, 300])

        base_response = gold_responses[fmt]

        # Inject required keywords if needed
        response = base_response + " " + " ".join(kw_pair)

        instruction = Instruction(
            prompt=f"Test instruction {i}: respond in {fmt or 'any'} format.",
            required_format=fmt,
            min_words=min_w,
            max_words=max_w,
            required_keywords=kw_pair,
            forbidden_keywords=[],
            language="en",
        )
        suite.append((instruction, response))

    return suite
