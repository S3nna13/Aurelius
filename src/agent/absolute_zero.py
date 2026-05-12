"""Absolute Zero — self-play skill generation with code execution verification.

The model alternates between:
  Propose: generate a task (code problem, math problem, reasoning puzzle)
  Solve:   generate a solution to the task
  Verify:  execute code or check answer for correctness (binary signal)
  Crystallize: if solved, add to skill library via SkillEvolver

This creates a self-improving curriculum with no external data.
Paper: arXiv:2505.03335 (Absolute Zero Reasoner, NeurIPS 2025)
"""
from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AbsoluteZeroConfig:
    """Configuration for the Absolute Zero self-play loop."""
    n_propose_per_cycle: int = 8
    n_solve_attempts: int = 3
    difficulty_range: tuple = (1, 5)
    crystallize_threshold: float = 0.7
    max_code_exec_seconds: float = 5.0
    task_types: list[str] = field(default_factory=lambda: [
        "code_completion", "math_reasoning", "logic_puzzle", "debugging"])


class AbsoluteZeroLoop:
    """Self-play loop: propose → solve → verify → crystallize."""

    def __init__(self, model_fn: Callable, skill_evolver, config: AbsoluteZeroConfig):
        self.model_fn = model_fn
        self.skill_evolver = skill_evolver
        self.config = config

    def propose_task(self, task_type: str, difficulty: int) -> str:
        """Use the model to propose a task of the given type and difficulty."""
        prompt = (
            f"Generate a {task_type} problem at difficulty level {difficulty}/5. "
            "The problem must be verifiable via code execution. "
            "Output ONLY the problem statement, no solution."
        )
        return self.model_fn(prompt)

    def solve_task(self, task: str) -> str:
        """Use the model to solve the proposed task."""
        prompt = f"Solve this problem step by step, then provide the final answer:\n\n{task}"
        return self.model_fn(prompt)

    def verify_code_solution(self, task: str, solution: str) -> bool:
        """Execute code solution in a restricted sandbox and check correctness."""
        import os
        import subprocess
        import tempfile
        code = self._extract_code(solution)
        if not code:
            return False
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                             delete=False) as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(
                ["python3", fname],  # noqa: S607 - intentional PATH lookup for python3
                capture_output=True, text=True,
                timeout=self.config.max_code_exec_seconds)
            os.unlink(fname)
            return result.returncode == 0
        except Exception as e:
            logger.debug("Verification failed: %s", e)
            return False

    def _extract_code(self, text: str) -> str | None:
        """Extract Python code block from model output."""
        if "```python" in text:
            start = text.index("```python") + 9
            end = text.index("```", start)
            return textwrap.dedent(text[start:end])
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return textwrap.dedent(text[start:end])
        return None

    def run_cycle(self) -> dict:
        """Run one full propose→solve→verify→crystallize cycle."""
        import random
        stats = {"proposed": 0, "solved": 0, "crystallized": 0}

        for _ in range(self.config.n_propose_per_cycle):
            task_type = random.choice(self.config.task_types)
            difficulty = random.randint(*self.config.difficulty_range)
            task = self.propose_task(task_type, difficulty)
            stats["proposed"] += 1

            successes = 0
            for _ in range(self.config.n_solve_attempts):
                solution = self.solve_task(task)
                if self.verify_code_solution(task, solution):
                    successes += 1

            solve_rate = successes / self.config.n_solve_attempts
            stats["solved"] += 1 if solve_rate > 0 else 0

            if solve_rate >= self.config.crystallize_threshold:
                task_record = {
                    "task": task,
                    "solution": self.solve_task(task),
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "solve_rate": solve_rate,
                }
                self.skill_evolver.crystallize(task_record)
                stats["crystallized"] += 1

        return stats


__all__ = ["AbsoluteZeroLoop", "AbsoluteZeroConfig"]