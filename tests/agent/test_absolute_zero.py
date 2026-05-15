"""Tests for Absolute Zero self-play skill generation."""

from __future__ import annotations

from src.agent.absolute_zero import AbsoluteZeroConfig, AbsoluteZeroLoop


class DummySkillEvolver:
    def __init__(self):
        self.crystallized = []

    def crystallize(self, task_record):
        self.crystallized.append(task_record)


class DummyModelFn:
    def __init__(self, response="def solution():\n    return 42"):
        self.response = response
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1
        return self.response


def test_extract_code_python():
    loop = AbsoluteZeroLoop(DummyModelFn(), DummySkillEvolver(), AbsoluteZeroConfig())
    text = "```python\ndef foo():\n    return 1\n```"
    code = loop._extract_code(text)
    assert code is not None
    assert "def foo" in code


def test_extract_code_no_python():
    loop = AbsoluteZeroLoop(DummyModelFn(), DummySkillEvolver(), AbsoluteZeroConfig())
    text = "no code here"
    code = loop._extract_code(text)
    assert code is None


def test_propose_task_returns_string():
    model_fn = DummyModelFn()
    evolver = DummySkillEvolver()
    loop = AbsoluteZeroLoop(model_fn, evolver, AbsoluteZeroConfig())
    task = loop.propose_task("code_completion", 3)
    assert isinstance(task, str)


def test_solve_task_returns_string():
    model_fn = DummyModelFn()
    evolver = DummySkillEvolver()
    loop = AbsoluteZeroLoop(model_fn, evolver, AbsoluteZeroConfig())
    solution = loop.solve_task("test task")
    assert isinstance(solution, str)


def test_verify_code_solution_rejects_non_code():
    model_fn = DummyModelFn(response="This is not code")
    loop = AbsoluteZeroLoop(model_fn, DummySkillEvolver(), AbsoluteZeroConfig())
    result = loop.verify_code_solution("task", "This is not code")
    assert result is False


def test_run_cycle_returns_stats():
    model_fn = DummyModelFn(response="```python\nx = 1\n```")
    evolver = DummySkillEvolver()
    config = AbsoluteZeroConfig(n_propose_per_cycle=2, n_solve_attempts=1)
    loop = AbsoluteZeroLoop(model_fn, evolver, config)
    stats = loop.run_cycle()
    assert "proposed" in stats
    assert "solved" in stats
    assert "crystallized" in stats
    assert stats["proposed"] == 2
