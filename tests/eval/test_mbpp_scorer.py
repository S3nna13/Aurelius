"""Unit tests for src.eval.mbpp_scorer."""

from __future__ import annotations

import pytest

from src.eval.mbpp_scorer import (
    MBPPProblem,
    MBPPSampleResult,
    pass_at_k,
    score_problems,
    score_single,
)


def _add_problem() -> MBPPProblem:
    return MBPPProblem(
        task_id=1,
        text="Return a + b.",
        code="def add(a, b):\n    return a + b\n",
        test_list=[
            "assert add(1, 2) == 3",
            "assert add(-5, 5) == 0",
        ],
        test_setup_code="",
    )


def test_score_single_correct_completion_passes():
    prob = _add_problem()
    res = score_single(prob, "def add(a, b):\n    return a + b\n", timeout_seconds=5.0)
    assert isinstance(res, MBPPSampleResult)
    assert res.passed is True
    assert res.error is None
    assert res.failed_test is None
    assert res.task_id == 1
    assert res.duration_ms >= 0.0


def test_score_single_wrong_completion_fails():
    prob = _add_problem()
    res = score_single(prob, "def add(a, b):\n    return a - b\n", timeout_seconds=5.0)
    assert res.passed is False
    assert res.error is not None
    # First assert "add(1,2)==3" fails (since -1 != 3).
    assert res.failed_test == "assert add(1, 2) == 3"


def test_score_single_infinite_loop_times_out():
    prob = MBPPProblem(
        task_id=2,
        text="spin",
        code="",
        test_list=["assert spin() == 0"],
    )
    completion = "def spin():\n    while True:\n        pass\n"
    res = score_single(prob, completion, timeout_seconds=1.0)
    assert res.passed is False
    assert res.error is not None
    assert "Timeout" in res.error


def test_score_single_syntax_error_captured():
    prob = _add_problem()
    res = score_single(prob, "def add(a, b):\n    return a + +\n", timeout_seconds=5.0)
    assert res.passed is False
    assert res.error is not None
    assert "SyntaxError" in res.error or "syntax" in res.error.lower()


def test_score_single_missing_import_captured():
    prob = MBPPProblem(
        task_id=3,
        text="sqrt",
        code="",
        test_list=["assert my_sqrt(4) == 2"],
    )
    # References math without importing it.
    completion = "def my_sqrt(x):\n    return math.isqrt(x)\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert res.error is not None
    assert "NameError" in res.error or "math" in res.error


def test_score_single_multi_test_second_fails():
    prob = MBPPProblem(
        task_id=4,
        text="double",
        code="",
        test_list=[
            "assert double(1) == 2",          # passes
            "assert double(3) == 999",        # fails
            "assert double(5) == 10",         # never reached
        ],
    )
    completion = "def double(x):\n    return x * 2\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert res.failed_test == "assert double(3) == 999"


def test_score_single_test_setup_code_is_prepended():
    prob = MBPPProblem(
        task_id=5,
        text="use setup global",
        code="",
        test_list=["assert magic() == SETUP_VALUE"],
        test_setup_code="SETUP_VALUE = 42",
    )
    completion = "def magic():\n    return 42\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is True, res.error


def test_score_single_empty_test_list_raises():
    prob = MBPPProblem(task_id=6, text="", code="", test_list=[])
    with pytest.raises(ValueError):
        score_single(prob, "x = 1\n", timeout_seconds=1.0)


def test_score_problems_aggregates_pass_at_1():
    prob = _add_problem()
    completions = [[
        "def add(a, b):\n    return a + b\n",   # correct
        "def add(a, b):\n    return a - b\n",   # wrong
    ]]
    out = score_problems(
        [prob], completions, k_values=[1], max_workers=2, timeout_seconds=5.0
    )
    assert out["n_problems"] == 1
    assert out["pass@1"] == pytest.approx(0.5)
    assert out["per_task"][0]["n_correct"] == 1
    assert out["per_task"][0]["n_samples"] == 2


def test_score_problems_pass_at_3():
    prob = _add_problem()
    completions = [[
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a - b\n",
        "def add(a, b):\n    return a + b\n",
    ]]
    out = score_problems(
        [prob], completions, k_values=[1, 3], max_workers=2, timeout_seconds=5.0
    )
    # With 2 correct / 3 samples, pass@3 must be 1.0 (every 3-subset is all 3).
    assert out["pass@3"] == pytest.approx(1.0)
    assert out["pass@1"] == pytest.approx(2.0 / 3.0)


def test_score_problems_empty_returns_zeros():
    out = score_problems([], [], k_values=[1])
    assert out == {
        "per_task": [],
        "n_problems": 0,
        "skipped_k": [],
        "pass@1": 0.0,
    }


def test_subprocess_isolation_filesystem(tmp_path):
    """Child subprocess writes to tmp_path; verify the scorer itself does not."""
    sentinel = tmp_path / "child_wrote_here.txt"
    prob = MBPPProblem(
        task_id=7,
        text="side effect",
        code="",
        test_list=["assert side_effect() == 0"],
    )
    completion = (
        "def side_effect():\n"
        f"    open({str(sentinel)!r}, 'w').write('hi')\n"
        "    return 0\n"
    )
    # Snapshot parent CWD files before.
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is True, res.error
    # Child did touch the file, but in the parent's configured tmp_path only.
    assert sentinel.exists()
    # Confirm the scorer itself did not create any other stray files in tmp_path.
    children = {p.name for p in tmp_path.iterdir()}
    assert children == {"child_wrote_here.txt"}


def test_determinism_same_inputs_same_outputs():
    prob = _add_problem()
    completion = "def add(a, b):\n    return a + b\n"
    r1 = score_single(prob, completion, timeout_seconds=5.0)
    r2 = score_single(prob, completion, timeout_seconds=5.0)
    assert r1.passed == r2.passed
    assert r1.error == r2.error
    assert r1.failed_test == r2.failed_test
    assert r1.task_id == r2.task_id


def test_pass_at_k_with_k_greater_than_n_raises():
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=1, k=5)


def test_score_problems_workers_equivalence():
    prob = _add_problem()
    completions = [[
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a - b\n",
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b\n",
    ]]
    o1 = score_problems(
        [prob], completions, k_values=[1], max_workers=1, timeout_seconds=5.0
    )
    o4 = score_problems(
        [prob], completions, k_values=[1], max_workers=4, timeout_seconds=5.0
    )
    assert o1["pass@1"] == pytest.approx(o4["pass@1"])
    assert o1["per_task"][0]["n_correct"] == o4["per_task"][0]["n_correct"]


def test_score_single_rejects_bad_timeout():
    prob = _add_problem()
    with pytest.raises(ValueError):
        score_single(prob, "def add(a,b):\n return a+b\n", timeout_seconds=0)
    with pytest.raises(ValueError):
        score_single(prob, "def add(a,b):\n return a+b\n", max_memory_mb=0)
