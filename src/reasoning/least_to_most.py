"""Least-to-Most Prompting: Zhou et al. 2022 'Least-to-Most Prompting Enables Complex Reasoning'."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class L2MConfig:
    max_subproblems: int = 5
    decompose_prompt: str = "Break this problem into simpler subproblems:\n"
    solve_prompt: str = "Given the above, solve:\n"


@dataclass(frozen=True)
class SubProblem:
    question: str
    answer: str = ""
    is_resolved: bool = False


@dataclass(frozen=True)
class L2MResult:
    final_answer: str
    subproblems: tuple[SubProblem, ...]
    n_steps: int


class LeastToMost:
    def __init__(self, config: L2MConfig | None = None) -> None:
        self.config = config or L2MConfig()

    def decompose(self, question: str, generate_fn: Callable[[str], str]) -> list[str]:
        response = generate_fn(self.config.decompose_prompt + question)
        subproblems: list[str] = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^\d+[.)]\s+", line):
                text = re.sub(r"^\d+[.)]\s+", "", line).strip()
                if text:
                    subproblems.append(text)
            elif re.match(r"^[-•]\s+", line):
                text = re.sub(r"^[-•]\s+", "", line).strip()
                if text:
                    subproblems.append(text)
        return subproblems[: self.config.max_subproblems]

    def solve_sequential(
        self,
        subproblems: list[str],
        generate_fn: Callable[[str, list[str]], str],
    ) -> list[SubProblem]:
        resolved: list[SubProblem] = []
        for sp in subproblems:
            prior_answers = [r.answer for r in resolved]
            answer = generate_fn(sp, prior_answers)
            resolved.append(SubProblem(question=sp, answer=answer, is_resolved=True))
        return resolved

    def run(
        self,
        question: str,
        generate_fn: Callable[[str], str],
        solve_fn: Callable[[str, list[str]], str] | None = None,
    ) -> L2MResult:
        subproblem_texts = self.decompose(question, generate_fn)
        effective_solve = solve_fn if solve_fn is not None else (lambda q, ctx: generate_fn(q))
        resolved = self.solve_sequential(subproblem_texts, effective_solve)
        final_answer = resolved[-1].answer if resolved else ""
        return L2MResult(
            final_answer=final_answer,
            subproblems=tuple(resolved),
            n_steps=len(resolved),
        )


L2M_REGISTRY: dict[str, type[LeastToMost]] = {"default": LeastToMost}
