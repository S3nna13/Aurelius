"""Best-of-N concurrent reasoning optimizer with ThinkingModeParser."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ThinkingBlock:
    content: str
    start_pos: int
    end_pos: int


@dataclass
class ParsedReasoning:
    thinking_blocks: list[ThinkingBlock]
    final_answer: str
    raw: str


class ThinkingModeParser:
    """Parses <think>...</think> blocks from model output (regex-based)."""

    THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def parse(self, text: str) -> ParsedReasoning:
        blocks: list[ThinkingBlock] = []
        for m in self.THINK_RE.finditer(text):
            blocks.append(
                ThinkingBlock(
                    content=m.group(1),
                    start_pos=m.start(),
                    end_pos=m.end(),
                )
            )

        if blocks:
            last_end = blocks[-1].end_pos
            final_answer = text[last_end:].strip()
        else:
            final_answer = text

        return ParsedReasoning(
            thinking_blocks=blocks,
            final_answer=final_answer,
            raw=text,
        )

    def strip_thinking(self, text: str) -> str:
        return self.THINK_RE.sub("", text).strip()


@dataclass
class CandidateResult:
    candidate_id: int
    reasoning: ParsedReasoning
    score: float
    selected: bool = False


def _default_scorer(reasoning: ParsedReasoning) -> float:
    return len(reasoning.final_answer) / (1 + len(reasoning.thinking_blocks))


class BestOfNOptimizer:
    """Run N candidates concurrently, score by answer quality, pick best."""

    def __init__(
        self,
        n: int = 5,
        scorer: Callable[[ParsedReasoning], float] | None = None,
    ) -> None:
        self.n = n
        self.scorer = scorer if scorer is not None else _default_scorer
        self._parser = ThinkingModeParser()

    def generate_candidates(self, prompt: str, n: int | None = None) -> list[str]:
        count = n if n is not None else self.n
        return [
            f"<think>Reasoning step {i}: analyzing {prompt[:50]}</think>\nAnswer {i}: result"
            for i in range(count)
        ]

    def score(self, reasoning: ParsedReasoning) -> float:
        return self.scorer(reasoning)

    def select_best(self, candidates: list[str]) -> CandidateResult:
        results: list[CandidateResult] = []
        for idx, raw in enumerate(candidates):
            parsed = self._parser.parse(raw)
            results.append(
                CandidateResult(
                    candidate_id=idx,
                    reasoning=parsed,
                    score=self.score(parsed),
                )
            )

        best = max(results, key=lambda r: r.score)
        best.selected = True
        return best

    def optimize(self, prompt: str) -> CandidateResult:
        candidates = self.generate_candidates(prompt)
        return self.select_best(candidates)


BEST_OF_N_REGISTRY: dict[str, type[BestOfNOptimizer]] = {"default": BestOfNOptimizer}
