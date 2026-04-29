"""Autoformalization loop — LLM + proof checker for theorem proving. (2601.03298)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Theorem:
    name: str
    statement: str
    proof: str = ""
    verified: bool = False


class Autoformalizer:
    """LLM + proof checker feedback loop for autoformalization.

    Inspired by Urban (2601.03298): 130k lines in 2 weeks for $100.
    Iterates: LLM generates formal proof → checker verifies → feedback → refine.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        checker_fn: Callable[[str], bool] | None = None,
    ):
        self.llm_fn = llm_fn or (lambda p: f"theorem {p[:20]} ...")
        self.checker_fn = checker_fn or (lambda p: True)
        self.theorems: list[Theorem] = []

    def formalize(self, informal: str, name: str = "") -> Theorem:
        prompt = f"Formalize this statement in set theory:\n{informal}"
        statement = self.llm_fn(prompt)
        theorem = Theorem(name=name or f"thm_{len(self.theorems)}", statement=statement)
        self.theorems.append(theorem)
        return theorem

    def prove(self, theorem: Theorem, max_attempts: int = 5) -> Theorem:
        for attempt in range(max_attempts):
            prompt = f"Prove: {theorem.statement}\nAttempt {attempt + 1}:"
            theorem.proof = self.llm_fn(prompt)
            theorem.verified = self.checker_fn(theorem.proof)
            if theorem.verified:
                break
        return theorem

    def loop(self, informal: str, name: str = "", max_iterations: int = 3) -> Theorem:
        theorem = self.formalize(informal, name)
        for _ in range(max_iterations):
            self.prove(theorem)
            if theorem.verified:
                break
            prompt = f"Revise the formal statement based on: {theorem.proof[:100]}"
            theorem.statement = self.llm_fn(prompt)
        return theorem
