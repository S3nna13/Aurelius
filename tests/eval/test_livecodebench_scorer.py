"""Unit tests for src.eval.livecodebench_scorer."""

from __future__ import annotations

import pytest

from src.eval.livecodebench_scorer import (
    LiveCodeProblem,
    LiveCodeResult,
    parse_date_filter,
    pass_at_k,
    score_problems,
    score_single,
)


def _echo_problem(task_id: str = "lcb_echo") -> LiveCodeProblem:
    return LiveCodeProblem(
        task_id=task_id,
        prompt="Read a line and print it.",
        starter_code="",
        test_cases=[("hello\n", "hello\n")],
        difficulty="easy",
        release_date="2025-06-01",
        contest_source="leetcode",
    )


def test_score_single_correct_passes():
    prob = _echo_problem()
    completion = "print(input())\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert isinstance(res, LiveCodeResult)
    assert res.passed is True
    assert res.error is None
    assert res.failed_case_idx is None
    assert res.task_id == "lcb_echo"
    assert res.duration_ms >= 0.0


def test_score_single_wrong_output_fails_with_case_idx():
    prob = _echo_problem()
    # Always prints "wrong" regardless of input.
    completion = "_ = input()\nprint('wrong')\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert res.error is not None
    assert res.failed_case_idx == 0


def test_score_single_multi_case_fails_on_third():
    prob = LiveCodeProblem(
        task_id="lcb_multi",
        prompt="Print input doubled (as int).",
        starter_code="",
        test_cases=[
            ("1\n", "2\n"),
            ("2\n", "4\n"),
            ("5\n", "10\n"),
        ],
        difficulty="easy",
    )
    # Completion works for first two but breaks on input == "5".
    completion = "x = int(input())\nif x == 5:\n    print(99)\nelse:\n    print(x * 2)\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert res.failed_case_idx == 2


def test_score_single_timeout_enforced():
    prob = LiveCodeProblem(
        task_id="lcb_spin",
        prompt="spin forever",
        starter_code="",
        test_cases=[("\n", "")],
    )
    completion = "while True:\n    pass\n"
    res = score_single(prob, completion, timeout_seconds=1.0)
    assert res.passed is False
    assert res.error is not None
    assert "Timeout" in res.error
    assert res.failed_case_idx == 0


def test_score_single_syntax_error_captured():
    prob = _echo_problem()
    completion = "def broken(:\n"  # syntax error
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is False
    assert res.error is not None
    assert res.failed_case_idx == 0


def _problem_with_date(task_id: str, date: str) -> LiveCodeProblem:
    return LiveCodeProblem(
        task_id=task_id,
        prompt="identity",
        starter_code="",
        test_cases=[("x\n", "x\n")],
        difficulty="easy",
        release_date=date,
        contest_source="leetcode",
    )


def test_date_filter_after_excludes_older():
    p_old = _problem_with_date("old", "2023-01-01")
    p_new = _problem_with_date("new", "2025-12-01")
    completions = [["print(input())\n"], ["print(input())\n"]]
    out = score_problems(
        [p_old, p_new],
        completions,
        k_values=[1],
        date_filter="after 2024-06-01",
        timeout_seconds=5.0,
    )
    assert out["n_problems"] == 1
    assert out["filtered_out"] == 1
    assert out["per_task"][0]["task_id"] == "new"


def test_date_filter_before_excludes_newer():
    p_old = _problem_with_date("old", "2023-01-01")
    p_new = _problem_with_date("new", "2025-12-01")
    completions = [["print(input())\n"], ["print(input())\n"]]
    out = score_problems(
        [p_old, p_new],
        completions,
        k_values=[1],
        date_filter="before 2024-06-01",
        timeout_seconds=5.0,
    )
    assert out["n_problems"] == 1
    assert out["filtered_out"] == 1
    assert out["per_task"][0]["task_id"] == "old"


def test_parse_date_filter_invalid():
    with pytest.raises(ValueError):
        parse_date_filter("during 2024-01-01")
    with pytest.raises(ValueError):
        parse_date_filter("after 2024/01/01")
    with pytest.raises(ValueError):
        parse_date_filter("after")


def test_per_difficulty_aggregation():
    p_easy = LiveCodeProblem(
        task_id="e1",
        prompt="",
        starter_code="",
        test_cases=[("1\n", "1\n")],
        difficulty="easy",
    )
    p_hard = LiveCodeProblem(
        task_id="h1",
        prompt="",
        starter_code="",
        test_cases=[("1\n", "1\n")],
        difficulty="hard",
    )
    completions = [
        ["print(input())\n"],  # easy: passes
        ["print('wrong')\n"],  # hard: fails
    ]
    out = score_problems([p_easy, p_hard], completions, k_values=[1], timeout_seconds=5.0)
    assert out["per_difficulty"]["easy"] == pytest.approx(1.0)
    assert out["per_difficulty"]["hard"] == pytest.approx(0.0)


def test_per_source_aggregation():
    p_lc = LiveCodeProblem(
        task_id="lc1",
        prompt="",
        starter_code="",
        test_cases=[("1\n", "1\n")],
        contest_source="leetcode",
    )
    p_cf = LiveCodeProblem(
        task_id="cf1",
        prompt="",
        starter_code="",
        test_cases=[("1\n", "1\n")],
        contest_source="codeforces",
    )
    completions = [
        ["print(input())\n"],
        ["print(input())\n"],
    ]
    out = score_problems([p_lc, p_cf], completions, k_values=[1], timeout_seconds=5.0)
    assert out["per_source"]["leetcode"] == pytest.approx(1.0)
    assert out["per_source"]["codeforces"] == pytest.approx(1.0)


def test_empty_problems_returns_zero():
    out = score_problems([], [], k_values=[1], timeout_seconds=5.0)
    assert out["n_problems"] == 0
    assert out["pass@1"] == 0.0
    assert out["per_task"] == []
    assert out["per_difficulty"] == {}
    assert out["per_source"] == {}


def test_pass_at_k_k_gt_n_raises():
    # Reused from humaneval.pass_at_k; k > n must raise.
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 5)


def test_determinism_same_inputs_same_result():
    prob = _echo_problem()
    completion = "print(input())\n"
    r1 = score_single(prob, completion, timeout_seconds=5.0)
    r2 = score_single(prob, completion, timeout_seconds=5.0)
    assert r1.passed == r2.passed
    assert r1.error == r2.error
    assert r1.failed_case_idx == r2.failed_case_idx


def test_starter_code_is_prepended():
    prob = LiveCodeProblem(
        task_id="starter",
        prompt="",
        starter_code="PREFIX = 'AUR_'\n",
        test_cases=[("x\n", "AUR_x\n")],
    )
    completion = "print(PREFIX + input())\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is True


def test_subprocess_isolation_via_tmp_path(tmp_path, monkeypatch):
    # Running in a temp cwd must not affect correctness: the scorer
    # uses subprocess with -I isolation, so cwd is irrelevant.
    monkeypatch.chdir(tmp_path)
    prob = _echo_problem()
    completion = "print(input())\n"
    res = score_single(prob, completion, timeout_seconds=5.0)
    assert res.passed is True


def test_empty_test_cases_raises():
    prob = LiveCodeProblem(
        task_id="empty",
        prompt="",
        starter_code="",
        test_cases=[],
    )
    with pytest.raises(ValueError):
        score_single(prob, "print(1)\n", timeout_seconds=5.0)


def test_score_problems_end_to_end_two():
    p1 = _echo_problem("p1")
    p2 = LiveCodeProblem(
        task_id="p2",
        prompt="",
        starter_code="",
        test_cases=[("3\n", "9\n")],
        difficulty="medium",
        contest_source="codeforces",
    )
    completions = [
        ["print(input())\n"],  # passes
        ["x=int(input()); print(x*x)\n"],  # 3*3=9 passes
    ]
    out = score_problems([p1, p2], completions, k_values=[1], timeout_seconds=5.0)
    assert out["n_problems"] == 2
    assert out["pass@1"] == pytest.approx(1.0)
