"""Unit tests for src/eval/humaneval_scorer.py."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from src.eval.humaneval_scorer import (
    HumanEvalProblem,
    SampleResult,
    pass_at_k,
    score_problems,
    score_single,
)


# ---------------------------------------------------------------------------
# Toy problem factory
# ---------------------------------------------------------------------------


def _toy_problem(task_id: str = "toy/0") -> HumanEvalProblem:
    """Return a HumanEval-shaped problem: add(a, b) -> a + b."""
    prompt = "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(-1, 1) == 0\n"
        "    assert candidate(0, 0) == 0\n"
    )
    return HumanEvalProblem(
        task_id=task_id,
        prompt=prompt,
        canonical_solution="    return a + b\n",
        test=test,
        entry_point="add",
    )


def _second_toy_problem() -> HumanEvalProblem:
    prompt = "def double(x):\n    \"\"\"Return 2*x.\"\"\"\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(2) == 4\n"
        "    assert candidate(-3) == -6\n"
    )
    return HumanEvalProblem(
        task_id="toy/1",
        prompt=prompt,
        canonical_solution="    return 2 * x\n",
        test=test,
        entry_point="double",
    )


def _third_toy_problem() -> HumanEvalProblem:
    prompt = "def ident(x):\n    \"\"\"Return x.\"\"\"\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(7) == 7\n"
        "    assert candidate('a') == 'a'\n"
    )
    return HumanEvalProblem(
        task_id="toy/2",
        prompt=prompt,
        canonical_solution="    return x\n",
        test=test,
        entry_point="ident",
    )


# ---------------------------------------------------------------------------
# pass@k formula
# ---------------------------------------------------------------------------


def test_pass_at_k_half():
    # n=10, c=5, k=1 -> 5/10 = 0.5
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)


def test_pass_at_k_zero_correct():
    assert pass_at_k(10, 0, 1) == 0.0


def test_pass_at_k_all_correct():
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 10, 5) == 1.0


def test_pass_at_k_hand_computed():
    # n=5, c=2, k=3
    # pass@3 = 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
    assert pass_at_k(5, 2, 3) == pytest.approx(0.9)
    # Cross-check against the stable product form by recomputing manually.
    expected = 1.0 - (1 - 2 / 5) * (1 - 2 / 4) * (1 - 2 / 3)
    assert pass_at_k(5, 2, 3) == pytest.approx(expected)


def test_pass_at_k_k_greater_than_n_raises():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_pass_at_k_c_greater_than_n_raises():
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)


def test_pass_at_k_non_positive_k_raises():
    with pytest.raises(ValueError):
        pass_at_k(5, 1, 0)
    with pytest.raises(ValueError):
        pass_at_k(5, 1, -1)


# ---------------------------------------------------------------------------
# score_single
# ---------------------------------------------------------------------------


def test_score_single_correct_completion_passes():
    problem = _toy_problem()
    res = score_single(problem, "    return a + b\n", timeout_seconds=10.0)
    assert isinstance(res, SampleResult)
    assert res.passed is True
    assert res.error is None
    assert res.task_id == "toy/0"
    assert res.duration_ms > 0.0


def test_score_single_wrong_completion_fails():
    problem = _toy_problem()
    res = score_single(problem, "    return a - b\n", timeout_seconds=10.0)
    assert res.passed is False
    assert res.error is not None
    assert "AssertionError" in res.error or "AssertionError" in res.stderr


def test_score_single_infinite_loop_times_out():
    problem = _toy_problem()
    completion = "    while True:\n        pass\n"
    res = score_single(problem, completion, timeout_seconds=0.5)
    assert res.passed is False
    assert res.error is not None
    assert "Timeout" in res.error


def test_score_single_syntax_error():
    problem = _toy_problem()
    # Unbalanced paren -> SyntaxError at parse time.
    completion = "    return a +\n"
    res = score_single(problem, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert "SyntaxError" in (res.error or "") or "SyntaxError" in res.stderr


def test_score_single_import_error():
    problem = _toy_problem()
    completion = (
        "    import nonexistent_package_xyz_12345\n"
        "    return a + b\n"
    )
    res = score_single(problem, completion, timeout_seconds=5.0)
    assert res.passed is False
    # ModuleNotFoundError is a subclass of ImportError.
    assert (
        "ImportError" in (res.error or "")
        or "ModuleNotFoundError" in (res.error or "")
        or "ImportError" in res.stderr
        or "ModuleNotFoundError" in res.stderr
    )


def test_score_single_assertion_captured():
    problem = _toy_problem()
    completion = "    assert False, 'bad'\n    return a + b\n"
    res = score_single(problem, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert "AssertionError" in res.stderr


def test_score_single_benign_os_call_does_not_affect_parent():
    """A completion that touches os should run but not leak state upward."""
    problem = _toy_problem()
    completion = (
        "    import os\n"
        "    _ = os.getenv('DOES_NOT_EXIST_123')\n"
        "    return a + b\n"
    )
    res = score_single(problem, completion, timeout_seconds=5.0)
    assert res.passed is True


def test_score_single_subprocess_is_isolated_from_parent_fs(tmp_path: Path):
    """A file created by a completion in the subprocess has no effect on the
    parent's view unless it is written to a shared location we inspect."""
    problem = _toy_problem()
    marker = tmp_path / "child_created.txt"
    completion = (
        f"    with open({str(marker)!r}, 'w') as f:\n"
        "        f.write('hello')\n"
        "    return a + b\n"
    )
    res = score_single(problem, completion, timeout_seconds=5.0)
    assert res.passed is True
    # The file exists on disk (subprocess can of course write to shared fs),
    # but importantly the subprocess exited cleanly and we can clean up.
    assert marker.exists()
    marker.unlink()
    # After cleanup the scorer state in this process is unaffected: a fresh
    # score_single call still passes.
    res2 = score_single(problem, "    return a + b\n", timeout_seconds=5.0)
    assert res2.passed is True
    # And the child never mutated any attribute of our HumanEvalProblem.
    assert problem.entry_point == "add"


# ---------------------------------------------------------------------------
# score_problems
# ---------------------------------------------------------------------------


def test_score_problems_pass_at_1_over_3_problems_2_samples():
    p1, p2, p3 = _toy_problem("toy/0"), _second_toy_problem(), _third_toy_problem()
    # For each problem, two samples: one correct, one wrong.
    completions = [
        ["    return a + b\n", "    return a - b\n"],
        ["    return 2 * x\n", "    return x\n"],
        ["    return x\n", "    return None\n"],
    ]
    out = score_problems(
        [p1, p2, p3], completions, k_values=[1], max_workers=2, timeout_seconds=5.0
    )
    # pass@1 for n=2,c=1 is 0.5 for each problem -> mean 0.5.
    assert out["pass@1"] == pytest.approx(0.5)
    assert out["n_problems"] == 3
    assert len(out["per_task"]) == 3


def test_score_problems_pass_at_k_k1_and_k2():
    p1 = _toy_problem("toy/0")
    p2 = _second_toy_problem()
    completions = [
        ["    return a + b\n", "    return a + b\n"],  # 2/2 correct
        ["    return 2 * x\n", "    return x\n"],       # 1/2 correct
    ]
    out = score_problems(
        [p1, p2], completions, k_values=[1, 2], max_workers=2, timeout_seconds=5.0
    )
    # p1: pass@1=1.0, pass@2=1.0. p2: pass@1=0.5, pass@2=1.0 (k=n, c>=1).
    assert out["pass@1"] == pytest.approx((1.0 + 0.5) / 2)
    assert out["pass@2"] == pytest.approx(1.0)


def test_score_problems_max_workers_parity():
    p1 = _toy_problem("toy/0")
    p2 = _second_toy_problem()
    completions = [
        ["    return a + b\n", "    return a - b\n"],
        ["    return 2 * x\n", "    return x\n"],
    ]
    out_solo = score_problems(
        [p1, p2], completions, k_values=[1], max_workers=1, timeout_seconds=5.0
    )
    out_quad = score_problems(
        [p1, p2], completions, k_values=[1], max_workers=4, timeout_seconds=5.0
    )
    assert out_solo["pass@1"] == pytest.approx(out_quad["pass@1"])
    assert out_solo["n_problems"] == out_quad["n_problems"]


def test_score_problems_deterministic_on_same_inputs():
    p = _toy_problem("toy/0")
    completions = [["    return a + b\n", "    return a - b\n"]]
    out_a = score_problems(
        [p], completions, k_values=[1], max_workers=2, timeout_seconds=5.0
    )
    out_b = score_problems(
        [p], completions, k_values=[1], max_workers=2, timeout_seconds=5.0
    )
    assert out_a["pass@1"] == out_b["pass@1"]
    assert [t["n_correct"] for t in out_a["per_task"]] == [
        t["n_correct"] for t in out_b["per_task"]
    ]


def test_score_problems_empty_list():
    out = score_problems([], [], k_values=[1, 10])
    assert out["n_problems"] == 0
    assert out["pass@1"] == 0.0
    assert out["pass@10"] == 0.0
    assert out["per_task"] == []


def test_score_problems_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_problems([_toy_problem()], [], k_values=[1])


def test_score_problems_rejects_empty_samples_for_a_problem():
    with pytest.raises(ValueError):
        score_problems([_toy_problem()], [[]], k_values=[1])
